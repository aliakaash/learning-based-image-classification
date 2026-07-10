"""
app.py
Streamlit front-end for the Image Classification System.
Lets the user upload an image, runs it through both the ANN and CNN models,
and shows predictions with confidence scores + a side-by-side comparison.

Run:
    streamlit run app.py
"""

import os
import json
import numpy as np
import cv2
from PIL import Image
import streamlit as st

from config import (
    CLASS_NAMES, ANN_MODEL_PATH, CNN_MODEL_PATH, OUTPUTS_DIR, IMG_SIZE,
)
from preprocessing import load_and_preprocess_image

st.set_page_config(page_title="Image Classification System", page_icon="🖼️", layout="wide")


@st.cache_resource
def load_models():
    import tensorflow as tf
    models = {}
    if os.path.exists(ANN_MODEL_PATH):
        models["ANN"] = tf.keras.models.load_model(ANN_MODEL_PATH)
    if os.path.exists(CNN_MODEL_PATH):
        models["CNN"] = tf.keras.models.load_model(CNN_MODEL_PATH)
    return models


@st.cache_data
def load_metrics_summary():
    path = os.path.join(OUTPUTS_DIR, "metrics_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def predict(model, img_array):
    batch = np.expand_dims(img_array, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], probs


def main():
    st.title("🖼️ Learning-Based Image Classification System")
    st.caption("TensorFlow • Keras • OpenCV • Streamlit")

    models = load_models()
    metrics = load_metrics_summary()

    if not models:
        st.error(
            "No trained models found in `models/`. Run `python data_prep.py` "
            "then `python train.py` first to generate `ann_model.h5` and `cnn_model.h5`."
        )
        st.stop()

    with st.sidebar:
        st.header("⚙️ Settings")
        selected_models = st.multiselect(
            "Models to use", list(models.keys()), default=list(models.keys())
        )
        st.markdown("---")
        st.header("📊 Classes")
        st.write(", ".join(CLASS_NAMES))

        if metrics:
            st.markdown("---")
            st.header("📈 Test Accuracy")
            for m in ["ann", "cnn"]:
                if m in metrics:
                    st.metric(m.upper(), f"{metrics[m]['test_accuracy']:.2%}")

    tab1, tab2 = st.tabs(["🔍 Classify Image", "📈 Model Performance"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png", "bmp"]
        )

        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            raw_array = np.array(pil_img)
            bgr_array = cv2.cvtColor(raw_array, cv2.COLOR_RGB2BGR)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(pil_img, use_container_width=True)
            with col2:
                st.subheader(f"Preprocessed ({IMG_SIZE[0]}x{IMG_SIZE[1]}, normalized)")
                processed = load_and_preprocess_image(bgr_array)
                st.image(processed, use_container_width=True, clamp=True)

            st.markdown("---")
            st.subheader("Predictions")

            if not selected_models:
                st.warning("Select at least one model from the sidebar.")
            else:
                cols = st.columns(len(selected_models))
                for col, model_name in zip(cols, selected_models):
                    model = models[model_name]
                    label, probs = predict(model, processed)
                    with col:
                        st.markdown(f"### {model_name}")
                        st.success(f"**Prediction: {label}**")
                        st.write(f"Confidence: {probs.max():.2%}")
                        prob_dict = {c: float(p) for c, p in zip(CLASS_NAMES, probs)}
                        st.bar_chart(prob_dict)

    with tab2:
        st.subheader("Training History & Evaluation")
        if metrics:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("ANN Test Accuracy", f"{metrics['ann']['test_accuracy']:.2%}")
            with c2:
                st.metric("CNN Test Accuracy", f"{metrics['cnn']['test_accuracy']:.2%}")

        for fname, caption in [
            ("model_comparison.png", "ANN vs CNN Test Accuracy"),
            ("ann_history.png", "ANN Training History"),
            ("cnn_history.png", "CNN Training History"),
            ("ann_confusion_matrix.png", "ANN Confusion Matrix"),
            ("cnn_confusion_matrix.png", "CNN Confusion Matrix"),
        ]:
            fpath = os.path.join(OUTPUTS_DIR, fname)
            if os.path.exists(fpath):
                st.image(fpath, caption=caption, use_container_width=True)

        report_path = os.path.join(OUTPUTS_DIR, "classification_reports.txt")
        if os.path.exists(report_path):
            with open(report_path) as f:
                st.text(f.read())


if __name__ == "__main__":
    main()
