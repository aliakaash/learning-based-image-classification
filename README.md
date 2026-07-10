# Learning-Based Image Classification System
TensorFlow • Keras • OpenCV • Streamlit

A complete image classification pipeline that trains **two models** (a plain
ANN and a CNN), compares them, and serves predictions through an interactive
Streamlit web app.

## Project Structure
```
image_classification_project/
├── config.py              # all paths & hyperparameters live here
├── data_prep.py            # generates a synthetic offline dataset (shapes)
├── preprocessing.py        # OpenCV-based load/resize/normalize utilities
├── models.py                # ANN and CNN architectures (Keras)
├── train.py                  # trains both models, saves plots & metrics
├── app.py                     # Streamlit app for interactive predictions
├── requirements.txt
├── data/
│   ├── train/<class_name>/*.jpg
│   └── test/<class_name>/*.jpg
├── models/                   # saved .h5 models land here after training
└── outputs/                   # plots, confusion matrices, reports, metrics.json
```

## 1. Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get a dataset
**Option A — use the built-in synthetic dataset (no download needed):**
```bash
python data_prep.py
```
This draws circles/squares/triangles/stars with OpenCV and creates
`data/train/` and `data/test/` automatically — good for testing the whole
pipeline end-to-end immediately.

**Option B — use your own real dataset:**
Just place your images like this (skip `data_prep.py`):
```
data/train/<class_name>/*.jpg
data/test/<class_name>/*.jpg
```
Then update `CLASS_NAMES` in `config.py` to match your folder names.

## 3. Train
```bash
python train.py
```
This will:
- Load & preprocess the dataset with OpenCV (resize, normalize, RGB convert)
- Train an **ANN** (Dense/fully-connected baseline)
- Train a **CNN** (with data augmentation) — expected to outperform the ANN
- Apply `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
- Save models to `models/ann_model.h5` and `models/cnn_model.h5`
- Save to `outputs/`:
  - `ann_history.png`, `cnn_history.png` — accuracy/loss curves
  - `ann_confusion_matrix.png`, `cnn_confusion_matrix.png`
  - `model_comparison.png` — ANN vs CNN accuracy bar chart
  - `classification_reports.txt` — precision/recall/F1 per class
  - `metrics_summary.json` — used by the Streamlit app sidebar

## 4. Run the app
```bash
streamlit run app.py
```
Upload an image and get:
- Side-by-side original vs. OpenCV-preprocessed image
- Predictions + confidence scores from both ANN and CNN
- A "Model Performance" tab with training curves, confusion matrices, and
  the full classification report

## Notes
- Default image size is 64×64 RGB (`config.IMG_SIZE`) — bump this up if you
  swap in a real, higher-resolution dataset.
- To add/remove classes, edit `CLASS_NAMES` in `config.py` and make sure
  your `data/train` / `data/test` folders match.
- All preprocessing (resize, color conversion, normalization) is centralized
  in `preprocessing.py` so training and the Streamlit app always stay consistent.
