#!/usr/bin/env python3
"""
LAB7: Vision Transformers
Starter template (exercise version)

Goals:
1. Load data
2. Build CNN (LAB5)
3. Build Vision Transformer
4. Train models
5. Perform experiments
6. Compare results
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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
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
import tensorflow_hub as hub

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 3


# ============================================================
# TODO 1: Load datasets
# ============================================================

def load_datasets(data_path, fraction=1.0):
    """
    TODO:
    - Load train/val/test datasets
    - Resize to 224x224
    - Normalize data
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path, validation_split=0.3, subset="training", seed=42,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='int'
    )
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_path, validation_split=0.3, subset="validation", seed=42,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='int'
    )

    val_ds = val_test_ds.take(len(val_test_ds) // 2)
    test_ds = val_test_ds.skip(len(val_test_ds) // 2)

    normalization_layer = layers.Rescaling(1. / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    if fraction < 1.0:
        train_ds = train_ds.take(max(1, int(len(train_ds) * fraction)))

    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE), test_ds.prefetch(tf.data.AUTOTUNE)


# ============================================================
# TODO 2: Augmentation
# ============================================================

def get_augmentation():
    """
    TODO:
    - Add augmentation:
        * flip
        * rotation
        * zoom
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ], name="data_augmentation")


# ============================================================
# TODO 3: CNN from LAB5
# ============================================================

def build_cnn(use_augmentation=False):
    """
    TODO:
    - Copy or recreate CNN from LAB5
    """
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = get_augmentation()(inputs) if use_augmentation else inputs
    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return models.Model(inputs, outputs, name="CNN_Baseline")


# ============================================================
# TODO 4: Patch extraction
# ============================================================

def create_patches(images, patch_size):
    """
    TODO:
    - Split image into patches
    """
    patches = tf.image.extract_patches(
        images=images, sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1], rates=[1, 1, 1, 1], padding='VALID'
    )
    return tf.reshape(patches, [tf.shape(images)[0], -1, patches.shape[-1]])


# ============================================================
# TODO 5: Patch encoding
# ============================================================

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        # TODO: embedding + positional encoding
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        # TODO
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)


# ============================================================
# TODO 6: Transformer block
# ============================================================

def transformer_block(x, num_heads, projection_dim):
    """
    TODO:
    - LayerNorm
    - MultiHeadAttention
    - MLP
    - Residual connections
    """
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
    x2 = layers.Add()([attn_output, x])

    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    x3_mlp = layers.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
    x3_mlp = layers.Dropout(0.1)(x3_mlp)
    x3_mlp = layers.Dense(projection_dim)(x3_mlp)
    x3_mlp = layers.Dropout(0.1)(x3_mlp)

    output = layers.Add()([x3_mlp, x2])
    return output


# ============================================================
# TODO 7: Vision Transformer
# ============================================================

def build_vit(use_augmentation=False):
    """
    TODO:
    - Input
    - Patch extraction
    - Encoder
    - Transformer blocks
    - Classifier
    """
    patch_size = 16
    num_patches = (IMG_SIZE // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_layers = 4

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = get_augmentation()(inputs) if use_augmentation else inputs

    x = layers.Lambda(lambda img: create_patches(img, patch_size))(x)
    x = PatchEncoder(num_patches, projection_dim)(x)

    for _ in range(transformer_layers):
        x = transformer_block(x, num_heads, projection_dim)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation=tf.nn.gelu)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    return models.Model(inputs, outputs, name="Vision_Transformer")


# ============================================================
# TODO 8: Compilation
# ============================================================

def compile_model(model):
    """
    TODO:
    - optimizer
    - loss
    - accuracy
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ============================================================
# TODO 9: Training
# ============================================================

def train_model(model, train_ds, val_ds):
    """
    TODO:
    - measure training time
    """
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)
    end = time.time()
    return history, end - start


# ============================================================
# TODO 10: Evaluation
# ============================================================

def evaluate_model(model, test_ds):
    """
    TODO:
    - accuracy
    - loss
    """
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    return accuracy, loss


# ============================================================
# TODO 11: Plots
# ============================================================

def plot_history(history, title="Model Training"):
    """
    TODO:
    - accuracy
    - loss
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('accuracy', []), label='Train Acc')
    plt.plot(history.history.get('val_accuracy', []), label='Val Acc')
    plt.title(f'{title} - Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('loss', []), label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)


# ============================================================
# TODO 12: Experiment A
# ============================================================

def experiment_small_data(train_ds, val_ds, test_ds):
    """
    TODO:
    - reduce dataset
    - CNN vs ViT
    """
    print("\n" + "=" * 50)
    print(" EXPERIMENT A: Reduced Dataset (40% Data)")
    print("=" * 50)

    cnn = compile_model(build_cnn(False))
    h_cnn, t_cnn = train_model(cnn, train_ds, val_ds)
    acc_cnn, loss_cnn = evaluate_model(cnn, test_ds)

    vit = compile_model(build_vit(False))
    h_vit, t_vit = train_model(vit, train_ds, val_ds)
    acc_vit, loss_vit = evaluate_model(vit, test_ds)

    plot_history(h_cnn, title="Experiment A: CNN (Reduced Data)")
    plot_history(h_vit, title="Experiment A: ViT (Reduced Data)")

    return {
        "cnn": {"acc": acc_cnn, "loss": loss_cnn, "time": t_cnn, "params": cnn.count_params()},
        "vit": {"acc": acc_vit, "loss": loss_vit, "time": t_vit, "params": vit.count_params()}
    }


# ============================================================
# TODO 13: Experiment B
# ============================================================

def experiment_full_aug(train_ds, val_ds, test_ds):
    """
    TODO:
    - augmentation
    """
    print("\n" + "=" * 50)
    print(" EXPERIMENT B: Full Dataset + Augmentation")
    print("=" * 50)

    cnn = compile_model(build_cnn(True))
    h_cnn, t_cnn = train_model(cnn, train_ds, val_ds)
    acc_cnn, loss_cnn = evaluate_model(cnn, test_ds)
    cnn.save("lab7_CNN_Full_Aug.keras")

    vit = compile_model(build_vit(True))
    h_vit, t_vit = train_model(vit, train_ds, val_ds)
    acc_vit, loss_vit = evaluate_model(vit, test_ds)
    vit.save("lab7_ViT_Full_Aug.keras")

    plot_history(h_cnn, title="Experiment B: CNN (Full + Aug)")
    plot_history(h_vit, title="Experiment B: ViT (Full + Aug)")

    return {
        "cnn": {"acc": acc_cnn, "loss": loss_cnn, "time": t_cnn, "params": cnn.count_params()},
        "vit": {"acc": acc_vit, "loss": loss_vit, "time": t_vit, "params": vit.count_params()}
    }


# ============================================================
# TODO 14: Experiment C (pretrained)
# ============================================================

def experiment_pretrained(train_ds, val_ds, test_ds):
    """
    TODO:
    - load pretrained ViT
    - modify classifier
    - fine-tuning
    """
    print("\n" + "=" * 50)
    print(" EXPERIMENT C: Pretrained ViT (ImageNet)")
    print("=" * 50)

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = get_augmentation()(inputs)
    hub_layer = hub.KerasLayer("https://tfhub.dev/sayakpaul/vit_s16_fe/1", trainable=False)(x)
    x = layers.Dense(128, activation='gelu')(hub_layer)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    pretrained_vit = compile_model(models.Model(inputs, outputs, name="Pretrained_ViT"))
    history, train_time = train_model(pretrained_vit, train_ds, val_ds)
    acc, loss = evaluate_model(pretrained_vit, test_ds)
    pretrained_vit.save("lab7_ViT_Pretrained.keras")

    plot_history(history, title="Experiment C: Pretrained ViT")

    return {
        "pretrained_vit": {"acc": acc, "loss": loss, "time": train_time, "params": pretrained_vit.count_params()}
    }


# ============================================================
# MAIN
# ============================================================

def main():
    # Activate Dual Logger
    sys.stdout = DualLogger("lab7_log.txt")

    # TODO:
    # - load datasets
    # - run experiments
    # - compare results
    data_path = "dataset"
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Dataset folder '{data_path}' not found!")
        return

    print("\n[1/3] Loading Full Dataset (for Exp B & C)...")
    train_full, val_full, test_full = load_datasets(data_path, fraction=1.0)

    print("[2/3] Loading Reduced Dataset (for Exp A)...")
    train_small, val_small, test_small = load_datasets(data_path, fraction=0.4)

    res_A = experiment_small_data(train_small, val_small, test_small)
    res_B = experiment_full_aug(train_full, val_full, test_full)

    try:
        res_C = experiment_pretrained(train_full, val_full, test_full)
    except Exception as e:
        print(f"\n[!] Error running Pretrained ViT experiment: {e}")
        res_C = {"pretrained_vit": {"acc": 0, "loss": 0, "time": 0, "params": 0}}

    print("\n\n" + "=" * 70)
    print("                     FINAL RESULTS TABLE")
    print("=" * 70)
    print(f"{'Model':<15} | {'Experiment':<12} | {'Accuracy':<10} | {'Loss':<8} | {'Time (s)':<8} | {'Params':<10}")
    print("-" * 70)

    def print_row(model, exp, res):
        print(
            f"{model:<15} | {exp:<12} | {res['acc']:.4f}     | {res['loss']:.4f}   | {res['time']:.2f}     | {res['params']}")

    print_row("CNN", "A (Reduced)", res_A["cnn"])
    print_row("ViT (Scratch)", "A (Reduced)", res_A["vit"])
    print_row("CNN", "B (Full+Aug)", res_B["cnn"])
    print_row("ViT (Scratch)", "B (Full+Aug)", res_B["vit"])
    print_row("ViT (Pretrain)", "C (Pretrain)", res_C["pretrained_vit"])
    print("=" * 70)
    print("\nLogs saved to 'lab7_log.txt'. Models saved as .keras files.")

    plt.show()


if __name__ == "__main__":
    main()
