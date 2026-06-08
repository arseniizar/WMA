#!/usr/bin/env python3
"""
LAB5: Convolutional Neural Networks
Exercise starter template

Task:
- Classify three fruits: banana, orange, lemon

Instructions:
- Complete all TODO sections
- First implement data loading
- Then build the model
- Finally train and test it
"""

import os
import sys

# --- LOGGER TO SAVE CONSOLE OUTPUT ---
class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Fix macOS specific crashes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# ============================================================
# TODO 1: Load dataset
# ============================================================

def load_data(dataset_path):
    """
    Loads images and labels from dataset.

    Expected structure:
    dataset/
        banana/
        orange/
        lemon/

    TODO:
    - Iterate through subdirectories (classes)
    - Load images using cv2.imread(...)
    - Resize images (e.g. 128x128)
    - Normalize pixel values (divide by 255)
    - Assign labels (e.g. 0,1,2)
    - Return:
        * X (images)
        * y (labels)
    """
    X, y = [], []
    label_map = {"banana": 0, "orange": 1, "lemon": 2}

    if not os.path.exists(dataset_path):
        return np.array(X), np.array(y)

    for class_name, label in label_map.items():
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            continue

        for file_name in os.listdir(class_path):
            if file_name.startswith('.'):
                continue
            file_path = os.path.join(class_path, file_name)
            img = cv2.imread(file_path)

            if img is not None:
                img = cv2.resize(img, (128, 128))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                X.append(img)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    if len(y) > 0:
        y = to_categorical(y, num_classes=3)

    return X, y


# ============================================================
# TODO 2: Data augmentation
# ============================================================

def create_augmentation():
    """
    Creates data augmentation pipeline.

    TODO:
    - Use ImageDataGenerator or tf.keras layers
    - Add transformations such as:
        * rotation
        * shift
        * flip
        * brightness change
    - Return augmentation object
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomBrightness(0.1)
    ], name="data_augmentation")


# ============================================================
# TODO 3: Build CNN model
# ============================================================

def build_model(input_shape, num_classes, augmentation=None):
    """
    Builds a convolutional neural network.

    TODO:
    - Add Conv2D layers
    - Add activation functions (e.g. ReLU)
    - Add MaxPooling layers
    - Add Flatten layer
    - Add Dense layers
    - Output layer:
        * neurons = number of classes
        * activation = softmax
    - Return model
    """
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))

    if augmentation:
        model.add(augmentation)

    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


# ============================================================
# TODO 4: Compile model
# ============================================================

def compile_model(model):
    """
    Compiles the model.

    TODO:
    - Choose loss function (e.g. categorical_crossentropy)
    - Choose optimizer (e.g. Adam)
    - Add metrics (e.g. accuracy)
    """
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# ============================================================
# TODO 5: Train model
# ============================================================

def train_model(model, X_train, y_train):
    """
    Trains the model.

    TODO:
    - Set:
        * number of epochs
        * batch size
    - Use model.fit(...)
    - Optionally:
        * validation split
        * early stopping
    - Return training history
    """
    X_t, X_val, y_t, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_t, y_t, epochs=30, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=2)
    return history


# ============================================================
# TODO 6: Test classification
# ============================================================

def classify_image(model, image_path):
    """
    Classifies a single image.

    TODO:
    - Load image
    - Resize to model input size
    - Normalize
    - Run model.predict(...)
    - Display result:
        * predicted class
        * probabilities
    """
    if not os.path.exists(image_path):
        print(f"[Warning] Test image '{image_path}' not found.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] Failed to load '{image_path}'.")
        return

    img_resized = cv2.resize(img, (128, 128))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0

    img_batch = np.expand_dims(img_normalized, axis=0)
    predictions = model.predict(img_batch)[0]

    class_idx = np.argmax(predictions)
    class_names_map = {0: "banana", 1: "orange", 2: "lemon"}
    predicted_class = class_names_map[class_idx]

    print(f"\n--- Classification Results for '{image_path}' ---")
    print(f"Predicted Class: {predicted_class.upper()}")
    for idx, prob in enumerate(predictions):
        print(f"  - {class_names_map[idx]}: {prob * 100:.2f}%")

    plt.imshow(img_rgb)
    plt.title(f"Predicted: {predicted_class.capitalize()}")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)


# ============================================================
# Main function
# ============================================================

def main():
    # Activate Dual Logger
    sys.stdout = DualLogger("lab5_log.txt")

    dataset_path = "dataset"

    # Parameters
    input_shape = (128, 128, 3)
    num_classes = 3

    # Load data
    print("Loading data...")
    X, y = load_data(dataset_path)

    if len(X) == 0:
        print("\nDATASET IS EMPTY!")
        return

    # Augmentation
    augmentation = create_augmentation()

    # Build model
    model = build_model(input_shape, num_classes, augmentation)

    # Compile model
    compile_model(model)

    # Show model structure
    model.summary()

    # Train model
    print("Starting training...")
    history = train_model(model, X, y)

    # TODO 7:
    # - Save trained model to file
    # - Plot training accuracy and loss
    model.save("model.keras")
    print("\nModel saved to 'model.keras'")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show(block=False)
    plt.pause(2)

    # Test example
    test_image_path = "test.jpg"

    if not os.path.exists(test_image_path):
        sample_dir = os.path.join(dataset_path, "banana")
        if os.path.exists(sample_dir):
            valid_files = [f for f in os.listdir(sample_dir) if not f.startswith('.')]
            if valid_files:
                test_image_path = os.path.join(sample_dir, valid_files[0])

    classify_image(model, test_image_path)

    print("\nScript completed. Logs saved to 'lab5_log.txt'.")
    plt.show()

if __name__ == "__main__":
    main()
