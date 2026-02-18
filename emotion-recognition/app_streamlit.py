# app_streamlit.py
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

import config as cfg


# ===================== Paths & Constants =====================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / cfg.BEST_MODEL_NAME
TEST_DIR = cfg.DATA_DIR / "test"

CLASS_NAMES = cfg.ACTIVE_CLASSES
IMG_SIZE = cfg.IMG_SIZE      # (H, W)
PIL_SIZE = cfg.PIL_SIZE      # (W, H)


# ===================== Load Model ===========================

@st.cache_resource
def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# ===================== Preprocessing ========================

def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(PIL_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_emotion(model: tf.keras.Model, img_arr: np.ndarray):
    preds = model.predict(img_arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx]), preds


# ===================== Quiz Helpers =========================

@st.cache_data
def load_quiz_images(limit_per_class: int = 300) -> List[Dict]:
    items = []
    for label in CLASS_NAMES:
        class_dir = TEST_DIR / label
        paths = list(class_dir.glob("*"))
        random.shuffle(paths)
        for p in paths[:limit_per_class]:
            img = Image.open(p).convert("RGB")
            items.append({"img": img, "label": label})
    random.shuffle(items)
    return items


# ===================== UI ================================

def main():
    st.set_page_config(page_title="Emotion Recognition", page_icon="😊")
    st.title("Emotion Recognition Demo")

    mode = st.sidebar.radio(
        "Mode",
        ["Single Image Prediction", "Quiz Mode"]
    )

    model = load_model()

    if mode == "Single Image Prediction":
        st.header("Single Image Prediction")

        file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if file:
            img = Image.open(file)
            st.image(img, use_container_width=True)

            arr = preprocess_pil_image(img)
            label, conf, probs = predict_emotion(model, arr)

            st.markdown(f"**Prediction:** {label}")
            st.markdown(f"**Confidence:** {conf:.3f}")

            st.table({
                "class": CLASS_NAMES,
                "probability": [float(p) for p in probs]
            })

    else:
        st.header("Emotion Quiz")

        quiz = load_quiz_images()
        idx = st.session_state.get("quiz_idx", 0) % len(quiz)
        item = quiz[idx]

        st.image(item["img"], use_container_width=True)

        guess = st.radio("Your guess:", CLASS_NAMES, horizontal=True)

        if st.button("Check"):
            st.session_state["quiz_idx"] = idx + 1
            if guess == item["label"]:
                st.success("Correct!")
            else:
                st.info(f"Correct label: {item['label']}")

if __name__ == "__main__":
    main()
