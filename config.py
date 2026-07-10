"""
config.py
Central configuration for the whole project.
Edit CLASS_NAMES and paths here if you plug in your own dataset.
"""

import os

# ----- Paths -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODELS_DIR = BASE_DIR
OUTPUTS_DIR = BASE_DIR

ANN_MODEL_PATH = os.path.join(MODELS_DIR, "ann_model.h5")
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_model.h5")

# ----- Dataset -----
# If you drop in your own dataset, just make sure it follows this structure:
#   data/train/<class_name>/*.jpg
#   data/test/<class_name>/*.jpg
CLASS_NAMES = ["circle", "square", "triangle", "star"]
IMG_SIZE = (64, 64)          # (height, width)
CHANNELS = 3

# ----- Synthetic dataset generation (used only if data/train is empty) -----
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_TEST = 60

# ----- Training -----
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15
RANDOM_SEED = 42

for _d in (DATA_DIR, TRAIN_DIR, TEST_DIR, MODELS_DIR, OUTPUTS_DIR):
    os.makedirs(_d, exist_ok=True)
