# LAB5 — Convolutional Neural Networks

Classifies images of fruits (bananas, oranges, lemons) using a custom-built Convolutional Neural Network (CNN) with
TensorFlow and Keras.

## Usage

Ensure you have a `dataset` folder in the same directory containing subfolders for each class (`banana`, `orange`,
`lemon`) with at least 450 images each, and optionally a `test.jpg` file.

```bash
python lab5.py
```

## How it works

The dataset is loaded and preprocessed (resized to 128x128, RGB, normalized) using OpenCV. To prevent overfitting, Keras
data augmentation layers (random flip, rotation, zoom, translation, brightness) are applied dynamically. The CNN
architecture extracts features via three blocks of `Conv2D` and `MaxPooling2D` layers, followed by flattening, a `Dense`
layer with `Dropout`, and a `Softmax` output layer for multi-class classification. The model is trained using the Adam
optimizer and Early Stopping, then evaluated and saved, finally outputting class probabilities for a test image.