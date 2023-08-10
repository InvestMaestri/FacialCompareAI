import os
import cv2
import sys
import math
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from tqdm import tqdm
from itertools import combinations
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import load_img, img_to_array, custom_object_scope
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Dropout, \
    BatchNormalization, Lambda

tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')
# -------------------------------------
# Flags
# -------------------------------------
debug = True

# Create a TensorFlow session
sess = tf.compat.v1.Session()

# Get a list of all available devices
devices = sess.list_devices()

# Check if any of the devices is a GPU
is_gpu_available = any(device.device_type == 'GPU' for device in devices)

if is_gpu_available:
    # Check if TensorFlow is using GPU
    print("GPU available:", tf.test.is_gpu_available())

    # Get the list of available GPUs
    gpus = tf.config.list_physical_devices("GPU")
    print("Available GPUs:", gpus)
else:
    print("GPU is not available. The model is running on CPU.")
    # Get the list of available GPUs
    gpus = tf.config.list_physical_devices("GPU")
    print("Available GPUs found:", gpus)

# Folder path containing the images
folder_path = "_2_2-Faces"

# List all image files in the folder
image_files = [file for file in os.listdir(folder_path) if file.endswith(".jpg") or file.endswith(".png")]

# Initialize the training data and labels
train_data = []
train_labels = []

# ---------------------------
# Create training data      |
# ---------------------------
print(f"Creating image pairs...")

# Heavy method for image pairs generates all possible combinations
# Define the desired percentage limit
limit_percent = 1

# Generate combinations of image files
combinations = list(combinations(image_files, 2))

# Calculate the number of pairs to be created based on the percentage limit
limit_count = int(len(combinations) * limit_percent)

# Randomly select a subset of combinations based on the limit count
selected_combinations = random.sample(combinations, limit_count)

# Define the progress bar
progress_bar = tqdm(total=len(selected_combinations), desc='Progress', unit='combination')

# Define batch size for incremental processing
batch_size = 1000

# Process image pairs in batches
for batch_start in range(0, len(selected_combinations), batch_size):
    batch_end = batch_start + batch_size
    batch_combinations = selected_combinations[batch_start:batch_end]

    # Process each image pair in the batch
    for pair in batch_combinations:
        image_file_1, image_file_2 = pair

        # Load and preprocess the first image
        image_1 = load_img(os.path.join(folder_path, image_file_1), target_size=(32, 32))
        image_1 = img_to_array(image_1)
        image_1 = image_1 / 255.0

        # Load and preprocess the second image
        image_2 = load_img(os.path.join(folder_path, image_file_2), target_size=(32, 32))
        image_2 = img_to_array(image_2)
        image_2 = image_2 / 255.0

        # Append the pair of images and labels to the training data
        train_data.append([image_1, image_2])

        if image_file_1.split("_")[0] == image_file_2.split("_")[0]:
            train_labels.append(1)  # Same pair label is 1
        else:
            train_labels.append(0)  # Different pair label is 0

        # Update the progress bar
        progress_bar.update(1)

    # Clear memory for the current batch
    del batch_combinations

# Close the progress bar
progress_bar.close()

# Convert the training data and labels to NumPy arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)

print(f"Pairs created.")
# Print the shapes of the training data and labels
if debug:
    print(f"Training data shape: {train_data.shape}")
    print(f"Training data length: {len(train_data)}")
    print(f"Training labels shape: {train_labels.shape}")

# Define the input shape of the images (width, height, channels)
input_shape = (32, 32, 3)  # Specify the dimensions of the input images


# Define the siamese network architecture
def create_siamese_network(input_shape):
    # Define the input layers for two images
    input_1 = Input(shape=input_shape)
    input_2 = Input(shape=input_shape)
    if debug:
        print(f'Input 1: {input_1}\nInput 2: {input_2}')

    # Shared convolutional layers
    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01))
    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01))
    maxpool = MaxPooling2D((2, 2))
    flatten = Flatten()
    dropout = Dropout(0.25)

    # Extract features from the first image
    x1 = conv1(input_1)
    x1 = maxpool(x1)
    x1 = BatchNormalization()(x1)
    x1 = conv2(x1)
    x1 = maxpool(x1)
    x1 = BatchNormalization()(x1)
    x1 = flatten(x1)
    x1 = dropout(x1)

    # Extract features from the second image
    x2 = conv1(input_2)
    x2 = maxpool(x2)
    x2 = BatchNormalization()(x2)
    x2 = conv2(x2)
    x2 = maxpool(x2)
    x2 = BatchNormalization()(x2)
    x2 = flatten(x2)
    x2 = dropout(x2)

    # Concatenate the feature vectors
    concatenated = concatenate([x1, x2])

    # Dense layers for classification
    dense1 = Dense(128, activation='relu')(concatenated)
    dense1 = BatchNormalization()(dense1)
    dense1 = dropout(dense1)
    dense2 = Dense(64, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = dropout(dense2)
    output = Dense(1, activation='linear')(dense2)  # Output linear for contrastive loss

    # Create the model
    model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=output)

    return model


# Create the siamese network
model = create_siamese_network(input_shape)

# Load previously trained weights
if os.path.exists("weights.best.hdf5"):
    print("\nPrevious weights found!\nLoading weights to resume training.\n")
    model.load_weights("weights.best.hdf5")

# Split the data into training and validation sets
val_split = 0.2  # 20% of the data will be used for validation
num_samples = len(train_data)
num_val_samples = int(val_split * num_samples)

# Randomly shuffle the data and labels
indices = np.random.permutation(num_samples)
train_data = train_data[indices]
train_labels = train_labels[indices]

# Split the data into training and validation sets
x_train = [train_data[:, 0], train_data[:, 1]]
y_train = train_labels
x_val = [train_data[:num_val_samples, 0], train_data[:num_val_samples, 1]]
y_val = train_labels[:num_val_samples]


# -------------------------------------
# Training Code                       |
# -------------------------------------
# Define the Contrastive Loss function
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    y_true = K.cast(y_true, dtype=K.floatx())  # Cast y_true to float data type
    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))


# Clear previous TensorFlow session
K.clear_session()

if is_gpu_available:
    # Set GPU device
    with tf.device("/GPU:0"):
        epochs = 10
        learning_rate = 0.1
        decay_rate = learning_rate / epochs
        momentum = 0.8
        sgd_optimizer = tf.keras.optimizers.SGD(
            learning_rate=(learning_rate * decay_rate),
            momentum=momentum,
            nesterov=True
        )
        model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        # with custom_object_scope({'contrastive_loss': contrastive_loss}):
        #     model.compile(optimizer=sgd_optimizer, loss=contrastive_loss, metrics=['accuracy'])

        # Print the model summary
        model.summary()

        # Callbacks
        save_path = "weights.best.hdf5"
        checkpoint = ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        es = EarlyStopping(monitor='val_loss', patience=500, verbose=1)
        callbacks_list = [checkpoint, es]

        # Train the model
        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks_list
        )

else:
    epochs = 10
    learning_rate = 0.1
    decay_rate = learning_rate / epochs
    momentum = 0.8
    sgd_optimizer = tf.keras.optimizers.SGD(
        learning_rate=(learning_rate * decay_rate),
        momentum=momentum,
        nesterov=True
    )
    model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # with custom_object_scope({'contrastive_loss': contrastive_loss}):
    #    model.compile(optimizer=sgd_optimizer, loss=contrastive_loss, metrics=['accuracy'])

    # Print the model summary
    model.summary()

    # Callbacks
    save_path = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    es = EarlyStopping(monitor='val_loss', patience=500, verbose=1)
    callbacks_list = [checkpoint, es]

    # Train the model
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=32,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks_list
    )

# Plot training loss and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training accuracy and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
