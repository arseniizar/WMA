# LAB7 — Vision Transformers (ViT)

Implementation and comparative analysis of Classical Convolutional Neural Networks (CNNs) versus Modern Vision Transformers (ViTs) using TensorFlow and Keras.

## Overview
This script demonstrates the difference between local filtering (CNNs) and global self-attention (Transformers). It features a fully custom-built Vision Transformer implementation built from scratch, alongside a classical baseline model.

Three distinct experiments are performed automatically:
- **Experiment A:** Compares the CNN and ViT models on a severely reduced dataset to demonstrate ViT's data hunger.
- **Experiment B:** Compares both models on a full-size dataset utilizing real-time Data Augmentation.
- **Experiment C:** Showcases the true power of Transformers by applying Transfer Learning to a massive Pre-trained Google ViT feature extractor via `tensorflow_hub`.

## Requirements
```bash
pip install tensorflow tensorflow-hub matplotlib numpy
```

## Running the Experiments
Place your dataset containing 3 object classes (at least 450 images each) into a directory named `dataset/` alongside the script. 

```bash
python lab7.py
```
*(Note: Experiment C requires an active internet connection on the first run to pull the pre-trained weights from tfhub.dev).*