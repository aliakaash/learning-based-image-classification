"""
preprocessing.py
OpenCV-based image loading & preprocessing utilities shared by train.py,
evaluate.py, and the Streamlit app (app.py).
"""

import os
import numpy as np
import cv2

from config import TRAIN_DIR, TEST_DIR, CLASS_NAMES, IMG_SIZE


def load_and_preprocess_image(path_or_array, img_size=IMG_SIZE):
    """
    Accepts either a file path (str) or an already-loaded BGR/RGB numpy array,
    and returns a normalized float32 array of shape (H, W, 3), values in [0, 1].
    """
    if isinstance(path_or_array, str):
        img = cv2.imread(path_or_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image at {path_or_array}")
    else:
        img = path_or_array
        if img.ndim == 2:  # grayscale -> 3 channel
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[-1] == 4:  # RGBA -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # keras models trained on RGB
    img = img.astype(np.float32) / 255.0
    return img


def load_dataset(split_dir):
    """
    Walks split_dir/<class_name>/*.jpg, preprocesses every image,
    and returns (X, y) as numpy arrays. y is integer-encoded per CLASS_NAMES.
    """
    X, y = [], []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Expected class folder not found: {class_dir}. "
                f"Check config.CLASS_NAMES matches your dataset folders."
            )
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            fpath = os.path.join(class_dir, fname)
            X.append(load_and_preprocess_image(fpath))
            y.append(label_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


def load_train_test():
    X_train, y_train = load_dataset(TRAIN_DIR)
    X_test, y_test = load_dataset(TEST_DIR)
    return X_train, y_train, X_test, y_test
