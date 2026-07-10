"""
models.py
Defines two architectures as required by the task:
  1. build_ann() -> a plain Artificial Neural Network (fully-connected / Dense)
  2. build_cnn() -> a Convolutional Neural Network

Both take flattened/2D image tensors of shape config.IMG_SIZE + (3,)
and output a softmax over len(config.CLASS_NAMES).
"""

from tensorflow.keras import layers, models, regularizers
from config import IMG_SIZE, CHANNELS, CLASS_NAMES, LEARNING_RATE
import tensorflow as tf


def build_ann():
    """Fully-connected ANN baseline (flattens the image first)."""
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], CHANNELS)
    n_classes = len(CLASS_NAMES)

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation="softmax"),
    ], name="ANN_Classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn():
    """Convolutional Neural Network — expected to outperform the ANN."""
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], CHANNELS)
    n_classes = len(CLASS_NAMES)

    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.15),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.15),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ], name="CNN_Classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    print(build_ann().summary())
    print(build_cnn().summary())
