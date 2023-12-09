from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
import numpy as np
import os
import nibabel as nib
from scipy import ndimage
import skfuzzy as fuzz

# --------------- Functions for Processing Scans ---------------------

# Function to read and load a volume
def read_nifti_file(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan

# Function to normalize the volume
def normalize(volume):
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

# Function to resize across the z-axis
def resize_volume(img):
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def apply_fuzzy_c_means(image, n_clusters=3):
    # Reshape the image to a 2D array of pixels
    reshaped_image = image.reshape(-1, 1)
    # Apply FCM
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        reshaped_image.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
    )
    # Find the cluster membership for each pixel
    cluster_membership = np.argmax(u, axis=0)
    # Reshape back to the original image shape
    clustered_image = cluster_membership.reshape(image.shape)
    return clustered_image

def process_scan_with_fcm(path):
    volume = read_nifti_file(path)
    volume = normalize(volume)
    volume = resize_volume(volume)
    volume = apply_fuzzy_c_means(volume)  # Apply FCM here
    return volume

class Specificity(tf.keras.metrics.Metric):
    def __init__(self, name='specificity', **kwargs):
        super(Specificity, self).__init__(name=name, **kwargs)
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_not(y_true)
        preds = tf.logical_not(y_pred)

        true_negatives = tf.reduce_sum(tf.cast(tf.logical_and(values, preds), self.dtype))
        false_positives = tf.reduce_sum(tf.cast(tf.logical_and(values, tf.logical_not(preds)), self.dtype))

        self.true_negatives.assign_add(true_negatives)
        self.false_positives.assign_add(false_positives)

    def result(self):
        specificity = self.true_negatives / (self.true_negatives + self.false_positives)
        return specificity

    def reset_states(self):
        self.true_negatives.assign(0)
        self.false_positives.assign(0)

class Sensitivity(tf.keras.metrics.Metric):
    def __init__(self, name='sensitivity', **kwargs):
        super(Sensitivity, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(y_true, y_pred), self.dtype))
        false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(y_true, tf.logical_not(y_pred)), self.dtype))

        self.true_positives.assign_add(true_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        sensitivity = self.true_positives / (self.true_positives + self.false_negatives)
        return sensitivity

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_negatives.assign(0)


# Load the saved model with custom objects
model = tf.keras.models.load_model(
    "/Users/ramazanmengi/PycharmProjects/DiseaseDetection/final_models/CT-Classifier-fuzzy-Rev-1-N2061223.h5",
    custom_objects={'Specificity': Specificity, 'Sensitivity': Sensitivity}
)

# Function to load and process scans from a folder
def load_scans(folder, label):
    file_paths = [os.path.join(folder, file) for file in os.listdir(folder)]
    scans = [process_scan_with_fcm(path) for path in file_paths]
    labels = [label] * len(scans)
    return scans, labels

# Load and process scans
normal_scans, normal_labels = load_scans("/Users/ramazanmengi/PycharmProjects/DiseaseDetection/CTData/normal", 0)
covid_scans, covid_labels = load_scans("/Users/ramazanmengi/PycharmProjects/DiseaseDetection/CTData/signs_of_covid", 1)

# Combine the scans and labels
all_scans = np.array(normal_scans + covid_scans)
all_labels = np.array(normal_labels + covid_labels)

# Expand dimensions for model input
all_scans = np.expand_dims(all_scans, axis=4)

# Predict on the dataset
predictions = model.predict(all_scans)
predicted_classes = np.argmax(predictions, axis=1)

# Calculate metrics
accuracy = accuracy_score(all_labels, predicted_classes)
precision = precision_score(all_labels, predicted_classes)
recall = recall_score(all_labels, predicted_classes)
f1 = f1_score(all_labels, predicted_classes)
roc_auc = roc_auc_score(all_labels, predictions[:, 1])

# Output the metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")
