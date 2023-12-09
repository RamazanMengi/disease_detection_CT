import tensorflow as tf
import nibabel as nib
from tensorflow import keras
import random
from scipy import ndimage
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, metrics
from tensorflow.keras.metrics import AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.math import confusion_matrix
import seaborn as sns


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

# Function to read, resize, and normalize a volume
def process_scan(path):
    volume = read_nifti_file(path)
    volume = normalize(volume)
    volume = resize_volume(volume)
    return volume

# -------------- Data Organization and Loading ---------------------


# Directory paths for different types of scans
folders = {
    "normal": "/Users/ramazanmengi/PycharmProjects/DiseaseDetection/CTData/normal",
    "signs_of_covid": "/Users/ramazanmengi/PycharmProjects/DiseaseDetection/CTData/signs_of_covid",
}

# Initialize dictionaries to hold scan paths and counts
scan_paths = {}
scan_counts = {}

# Populate scan paths and print counts for each type of scan
for label, folder in folders.items():
    paths = [os.path.join(os.getcwd(), folder, x) for x in os.listdir(folder)]
    scan_paths[label] = paths
    scan_counts[label] = len(paths)
    print(f"CT scans with {label} lung tissue: {len(paths)}")

# Initialize dictionaries to hold processed scans and labels
scans = {}
labels = {}

# Process the scans and set their labels
for label, paths in scan_paths.items():
    scans[label] = np.array([process_scan(path) for path in paths])
    labels[label] = np.array([label for _ in range(len(paths))])

# Categories in your dataset
categories = ['normal', 'signs_of_covid']

# Create a mapping of categories to numerical labels
label_mapping = {category: idx for idx, category in enumerate(categories)}

# Convert string labels to numerical labels
numerical_labels = {label: np.array([label_mapping[l] for l in labels[label]]) for label in labels}

# Separate the scans and labels for each category
normal_scans = scans['normal']
normal_labels = numerical_labels['normal']
signs_of_covid_scans = scans['signs_of_covid']
signs_of_covid_labels = numerical_labels['signs_of_covid']

# Calculate the index for a 70-30 split
split_idx = lambda x: int(0.8 * len(x))

# Combine the scans and labels for each category for training
x_train = np.concatenate(
    [category_scans[:split_idx(category_scans)] for category_scans in [normal_scans, signs_of_covid_scans]],
    axis=0
)
y_train = np.concatenate(
    [category_labels[:split_idx(category_labels)] for category_labels in [normal_labels, signs_of_covid_labels]],
    axis=0
)
# Combine the scans and labels for each category for validation
x_val = np.concatenate(
    [category_scans[split_idx(category_scans):] for category_scans in [normal_scans, signs_of_covid_scans]],
    axis=0
)
y_val = np.concatenate(
    [category_labels[split_idx(category_labels):] for category_labels in [normal_labels, signs_of_covid_labels]],
    axis=0
)

# Correct the number of classes for one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=len(categories))
y_val = keras.utils.to_categorical(y_val, num_classes=len(categories))

print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)

@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)


data = train_dataset.take(1)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimension of the CT scan is:", image.shape)
plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")


def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of CT slices."""
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    fig, axarr = plt.subplots(rows_data, columns_data, figsize=(fig_width, fig_height),
                              gridspec_kw={"height_ratios": heights})

    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the CT scan.
plot_slices(4, 10, 128, 128, image[:, :, :40])

def get_model(width=128, height=128, depth=64, num_classes=2):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="3dcnn")

    # Define the model.
    return model

# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()

# Compile model.
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,  # Start with a lower learning rate
    decay_steps=10000,  # Adjust decay steps
    decay_rate=0.9,  # Adjust decay rate
    staircase=True
)


# Make predictions on the validation dataset
val_predictions = model.predict(x_val)
val_predictions = np.argmax(val_predictions, axis=1)
val_true = np.argmax(y_val, axis=1)

# Calculate the confusion matrix
cm = confusion_matrix(val_true, val_predictions)

# Visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Define custom metrics for specificity and sensitivity
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

# Include these custom metrics in your model's compilation step
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.legacy.Adam(learning_rate=lr_schedule),
    metrics=[
        "acc",
        metrics.Precision(),
        metrics.Recall(),
        AUC(),
        Specificity(),
        Sensitivity(),
    ],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "final_models/CT-Classifier-Rev-2.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

# Train the model, doing validation at the end of each epoch
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=100,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])


# Load best weights.
model.load_weights("final_models/CT-Classifier-Rev-2.h5")
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ["normal", "abnormal"]

def plot_training_history(history):
    """Plot accuracy and loss graphs for the training history."""
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy NN')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss NN')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_training_history(history)

# Make predictions on the validation dataset
val_predictions = model.predict(x_val)
val_predictions = np.argmax(val_predictions, axis=1)
val_true = np.argmax(y_val, axis=1)

# Calculate the confusion matrix
cm = confusion_matrix(val_true, val_predictions)

# Visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted labels NN')
plt.ylabel('True labels')
plt.show()




