import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import ndimage
import tensorflow as tf
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
    "/Users/ramazanmengi/PycharmProjects/DiseaseDetection/final_models/CT-Classifier-fuzzy-Rev-2-N3.h5",
    custom_objects={'Specificity': Specificity, 'Sensitivity': Sensitivity}
)

# Path to a new scan
new_scan_path = "/Users/ramazanmengi/PycharmProjects/DiseaseDetection/CTData/normal/study_0001.nii.gz"

# Preprocess the new scan with fuzzy logic
new_scan = process_scan_with_fcm(new_scan_path)

# Expand dimensions to fit into the model input shape
new_scan = np.expand_dims(new_scan, axis=0)
new_scan = np.expand_dims(new_scan, axis=4)  # Model expects 5D input

# Predict using the model
prediction = model.predict(new_scan)
predicted_class = np.argmax(prediction, axis=1)

# Extract the confidence (probability) of the predicted class
confidence = prediction[0, predicted_class[0]]

# Output the prediction and confidence
class_names = ["normal", "signs_of_covid"]
print("The model predicts this scan as:", class_names[predicted_class[0]], "with a confidence of:", confidence)

# (Optional) Visualize the scan
plt.imshow(new_scan[0, :, :, 30, 0], cmap='gray')  # Adjust the index 30 as needed
plt.title("Slice of the CT Scan")
plt.show()
