# LAB5 — Convolutional Neural Networks

Classifies images of fruits (bananas, oranges, lemons) using a custom-built Convolutional Neural Network (CNN) with
TensorFlow and Keras.

## Demo video

<img width="800" height="424" alt="ezgif-57e090c59b64259a" src="https://github.com/user-attachments/assets/c48bd19f-24f8-4f43-bd29-e6ac3cd0fa37" />


## Results

<img width="642" height="537" alt="Screenshot 2026-06-09 at 20 42 18" src="https://github.com/user-attachments/assets/733b921b-fa04-422d-91c1-c06df06353d4" />
<img width="1197" height="560" alt="Screenshot 2026-06-09 at 20 42 12" src="https://github.com/user-attachments/assets/0a01cc8c-f603-4667-93da-887b618d829f" />

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
