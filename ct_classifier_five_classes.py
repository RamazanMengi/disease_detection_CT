import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import ndimage

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

import os
import numpy as np

# Directory paths for different types of scans
folders = {
    "normal": "data/normal",
    "signs_of_covid": "data/signs_of_covid",
    "adenocarcinoma": "data/adenocarcinoma",
    "small_cell_carcinoma": "data/small_cell_carcinoma",
    "squamous_cell_carcinoma": "data/squamous_cell_carcinoma",
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
categories = [
    'normal', 'signs_of_covid', 'adenocarcinoma',
    'squamous_cell_carcinoma', 'small_cell_carcinoma'
]

# Create a mapping of categories to numerical labels
label_mapping = {category: idx for idx, category in enumerate(categories)}

# Convert string labels to numerical labels
numerical_labels = {label: np.array([label_mapping[l] for l in labels[label]]) for label in labels}

# Separate the scans and labels for each category
normal_scans = scans['normal']
normal_labels = numerical_labels['normal']
signs_of_covid_scans = scans['signs_of_covid']
signs_of_covid_labels = numerical_labels['signs_of_covid']
adenocarcinoma_scans = scans['adenocarcinoma']
adenocarcinoma_labels = numerical_labels['adenocarcinoma']
squamous_cell_carcinoma_scans = scans['squamous_cell_carcinoma']
squamous_cell_carcinoma_labels = numerical_labels['squamous_cell_carcinoma']
small_cell_carcinoma_scans = scans['small_cell_carcinoma']
small_cell_carcinoma_labels = numerical_labels['small_cell_carcinoma']


# Calculate the index for a 70-30 split
split_idx = lambda x: int(0.85 * len(x))

# Combine the scans and labels for each category for training
x_train = np.concatenate(
    [category_scans[:split_idx(category_scans)] for category_scans in [normal_scans, signs_of_covid_scans, adenocarcinoma_scans, squamous_cell_carcinoma_scans, small_cell_carcinoma_scans]],
    axis=0
)
y_train = np.concatenate(
    [category_labels[:split_idx(category_labels)] for category_labels in [normal_labels, signs_of_covid_labels, adenocarcinoma_labels, squamous_cell_carcinoma_labels, small_cell_carcinoma_labels]],
    axis=0
)
# Combine the scans and labels for each category for validation
x_val = np.concatenate(
    [category_scans[split_idx(category_scans):] for category_scans in [normal_scans, signs_of_covid_scans, adenocarcinoma_scans, squamous_cell_carcinoma_scans, small_cell_carcinoma_scans]],
    axis=0
)
y_val = np.concatenate(
    [category_labels[split_idx(category_labels):] for category_labels in [normal_labels, signs_of_covid_labels, adenocarcinoma_labels, squamous_cell_carcinoma_labels, small_cell_carcinoma_labels]],
    axis=0
)

# Correct the number of classes for one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=len(categories))
y_val = keras.utils.to_categorical(y_val, num_classes=len(categories))

# ...

print(f"Number of samples in train and validation are {x_train.shape[0]} and {x_val.shape[0]}.")


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
    """Plot a grid of CT slices to visually inspect their orientation."""
    # Assuming data is a single volume (3D array)
    fig, ax = plt.subplots(num_rows, num_columns, figsize=(10, 10))
    for i in range(num_rows):
        for j in range(num_columns):
            # Assuming 64 slices per volume, change if needed
            ind = j + i * num_columns
            if ind < data.shape[-1]:  # ensure index is within bounds
                ax[i, j].imshow(data[:, :, ind], cmap='gray')
                ax[i, j].axis('off')
    plt.tight_layout()
    plt.show()

# Test the function with the first scan
# plot_slices(4, 10, 128, 128, scans['normal'][0])

def standardize_orientation(volume):
    """
    Ensure that the volume is oriented in the same way for all scans.
    You may need to adjust the logic depending on how you define 'standard' orientation.
    This function is a placeholder for the concept and may require specific adjustments.
    """
    # Placeholder: you might need to implement a check here to determine the current orientation
    # and then decide which rotation to apply. This could be based on the metadata or heuristics.
    # For demonstration, let's assume we rotate all volumes by 90 degrees around the z-axis.
    standard_volume = ndimage.rotate(volume, 90, axes=(0, 1), reshape=False)
    return standard_volume

# Apply the standardization to all scans
for label in scans:
    for i in range(len(scans[label])):
        scans[label][i] = standardize_orientation(scans[label][i])

# Now we can re-plot to check the orientations
# plot_slices(4, 10, 128, 128, scans['normal'][:5])

def get_model(width=128, height=128, depth=64, num_classes=5):
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
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.legacy.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 100
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
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
model.load_weights("3d_image_classification.h5")
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ['normal', 'signs_of_covid', 'adenocarcinoma', 'squamous_cell_carcinoma', 'small_cell_carcinoma']
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]

for score, name in zip(prediction, class_names):
    print("This model is %.2f percent confident that the CT scan shows %s" % (100 * score, name))

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming y_val are your true labels and model_predictions are your model's predictions
model_predictions = model.predict(x_val).argmax(axis=1)
true_labels = y_val.argmax(axis=1)
matrix = confusion_matrix(true_labels, model_predictions)

sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import classification_report

print(classification_report(true_labels, model_predictions, target_names=class_names))

from sklearn.metrics import roc_curve, auc
from itertools import cycle

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
y_val_categorical = keras.utils.to_categorical(true_labels, num_classes=len(class_names))
for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(y_val_categorical[:, i], model.predict(x_val)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
colors = cycle(['blue', 'red', 'green', 'yellow', 'purple'])
for i, color in zip(range(len(class_names)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(len(class_names)):
    precision[i], recall[i], _ = precision_recall_curve(y_val_categorical[:, i], model.predict(x_val)[:, i])
    average_precision[i] = average_precision_score(y_val_categorical[:, i], model.predict(x_val)[:, i])

# Plot the precision-recall curves
for i, color in zip(range(len(class_names)), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label='Precision-Recall curve of class {0} (area = {1:0.2f})'
                   ''.format(class_names[i], average_precision[i]))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left")
plt.title('Multi-class Precision-Recall curve')
plt.show()


history = model.fit(...)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['acc'], label='Train')
plt.plot(history.history['val_acc'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.show()
