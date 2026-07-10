"""
train.py
Trains both the ANN and CNN models, applies optimization callbacks
(EarlyStopping, ReduceLROnPlateau, ModelCheckpoint), evaluates on the
held-out test set, and saves:
  - models/ann_model.h5, models/cnn_model.h5
  - outputs/ann_history.png, outputs/cnn_history.png
  - outputs/*_confusion_matrix.png
  - outputs/classification_reports.txt
  - outputs/model_comparison.png

Run:
    python data_prep.py     # only needed once, or if you haven't added your own data
    python train.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    CLASS_NAMES, MODELS_DIR, OUTPUTS_DIR, ANN_MODEL_PATH, CNN_MODEL_PATH,
    BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, RANDOM_SEED,
)
from preprocessing import load_train_test
from models import build_ann, build_cnn

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def get_callbacks(checkpoint_path):
    return [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ]


def plot_history(history, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title(f"{title} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title(f"{title} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def train_one_model(name, build_fn, checkpoint_path, X_train, y_train, X_test, y_test,
                     use_augmentation=False):
    print(f"\n{'='*60}\nTraining {name}\n{'='*60}")
    model = build_fn()

    n_val = int(len(X_train) * VALIDATION_SPLIT)
    idx = np.random.permutation(len(X_train))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    callbacks = get_callbacks(checkpoint_path)

    if use_augmentation:
        datagen = ImageDataGenerator(
            rotation_range=8, width_shift_range=0.05, height_shift_range=0.05,
            zoom_range=0.05,
        )
        history = model.fit(
            datagen.flow(X_tr, y_tr, batch_size=BATCH_SIZE, seed=RANDOM_SEED),
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=2,
        )
    else:
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=2,
        )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{name} Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    plot_history(history, name, os.path.join(OUTPUTS_DIR, f"{name.lower()}_history.png"))
    plot_confusion(y_test, y_pred, f"{name} Confusion Matrix",
                   os.path.join(OUTPUTS_DIR, f"{name.lower()}_confusion_matrix.png"))

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    return model, test_acc, test_loss, report


def main():
    print("Loading & preprocessing dataset...")
    X_train, y_train, X_test, y_test = load_train_test()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    ann_model, ann_acc, ann_loss, ann_report = train_one_model(
        "ANN", build_ann, ANN_MODEL_PATH, X_train, y_train, X_test, y_test,
        use_augmentation=False,
    )

    cnn_model, cnn_acc, cnn_loss, cnn_report = train_one_model(
        "CNN", build_cnn, CNN_MODEL_PATH, X_train, y_train, X_test, y_test,
        use_augmentation=True,
    )

    # Save classification reports
    with open(os.path.join(OUTPUTS_DIR, "classification_reports.txt"), "w") as f:
        f.write("ANN Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(ann_report + "\n\n")
        f.write("CNN Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(cnn_report + "\n")

    # Model comparison bar chart
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(["ANN", "CNN"], [ann_acc, cnn_acc], color=["#4C72B0", "#55A868"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("ANN vs CNN — Test Accuracy Comparison")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                 f"{b.get_height():.2%}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)

    # Save summary metrics as JSON for the Streamlit app / report
    summary = {
        "ann": {"test_accuracy": float(ann_acc), "test_loss": float(ann_loss)},
        "cnn": {"test_accuracy": float(cnn_acc), "test_loss": float(cnn_loss)},
        "class_names": CLASS_NAMES,
    }
    with open(os.path.join(OUTPUTS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAll done. Models saved to models/, plots & metrics saved to outputs/.")
    print(f"ANN test accuracy: {ann_acc:.4f}")
    print(f"CNN test accuracy: {cnn_acc:.4f}")


if __name__ == "__main__":
    main()
