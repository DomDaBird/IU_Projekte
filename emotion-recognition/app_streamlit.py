"""
Streamlit application for the Emotion Recognition project.

Modes:
- Single Image Prediction
- Quiz Mode

This UI is designed for a company scenario where a marketing department
needs a simple, documented demo application.

Run:
  streamlit run app_streamlit.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

import config as cfg

# ============================================================
# Constants / Paths
# ============================================================

MODEL_PATH = cfg.MODELS_DIR / cfg.BEST_MODEL_NAME
TEST_DIR = cfg.DATA_DIR / "test"


# ============================================================
# Data structures
# ============================================================


@dataclass(frozen=True)
class QuizItem:
    image_path: Path
    label: str


# ============================================================
# Caching: model + quiz index
# ============================================================


@st.cache_resource
def load_model() -> tf.keras.Model:
    """
    Load the trained model once per app session.
    The model includes preprocessing in-graph (see model.py).
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}\n"
            "Expected a trained model at models/best_model.keras."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def build_quiz_items(
    limit_per_class: int = 300, seed: int = cfg.SEED
) -> List[QuizItem]:
    """
    Build quiz item list from test dataset folder.

    Args:
        limit_per_class: max images per class to include (avoids huge memory usage)
        seed: deterministic shuffling seed

    Returns:
        list of QuizItem
    """
    rng = random.Random(seed)
    items: List[QuizItem] = []

    for label in cfg.ACTIVE_CLASSES:
        class_dir = TEST_DIR / label
        if not class_dir.exists():
            # quiz can still run with other classes; show warning later
            continue

        paths = [
            p
            for p in class_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ]
        rng.shuffle(paths)
        for p in paths[:limit_per_class]:
            items.append(QuizItem(image_path=p, label=label))

    rng.shuffle(items)
    return items


# ============================================================
# Inference helpers
# ============================================================


def preprocess_pil(img: Image.Image) -> np.ndarray:
    """
    Convert PIL image to model input array.

    The model contains preprocessing, so we only resize and return uint8.
    """
    img = img.convert("RGB").resize(cfg.PIL_SIZE)
    arr = np.array(img, dtype=np.uint8)
    return np.expand_dims(arr, axis=0)


def predict(
    model: tf.keras.Model, img_arr: np.ndarray
) -> Tuple[str, float, np.ndarray]:
    """Return (pred_label, confidence, probs)."""
    probs = model.predict(img_arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return cfg.ACTIVE_CLASSES[idx], float(probs[idx]), probs


def probs_table(probs: np.ndarray) -> List[Dict[str, object]]:
    """Return a table-friendly list for Streamlit."""
    return [
        {"class": c, "probability": float(p)} for c, p in zip(cfg.ACTIVE_CLASSES, probs)
    ]


# ============================================================
# UI sections
# ============================================================


def ui_header() -> None:
    st.set_page_config(
        page_title="Emotion Recognition", page_icon="😊", layout="centered"
    )
    st.title("Emotion Recognition Demo")
    st.caption("AI Facial Emotion Recognition Application via Streamlit")


def ui_sidebar() -> str:
    st.sidebar.header("Navigation")
    mode = st.sidebar.radio(
        "Mode", ["Single Prediction", "Quiz Mode", "About"], index=0
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Company scenario: built for a marketing demo and maintainable handover."
    )
    return mode


def ui_about() -> None:
    st.header("About")
    st.write(
        "This application predicts facial emotions from images using a trained deep learning model "
        "based on transfer learning (EfficientNetB0)."
    )
    st.subheader("How to run")
    st.code("streamlit run app_streamlit.py", language="bash")
    st.subheader("Model file")
    st.code(str(MODEL_PATH))
    st.subheader("Classes")
    st.write(", ".join(cfg.ACTIVE_CLASSES))


def ui_single_prediction(model: tf.keras.Model) -> None:
    st.header("Single Image Prediction")
    st.write(
        "Upload a facial image (JPG/PNG). The model returns the predicted emotion and probabilities."
    )

    file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if not file:
        st.info("Upload an image to start.")
        return

    img = Image.open(file)
    st.image(img, use_container_width=True)

    x = preprocess_pil(img)
    label, conf, probs = predict(model, x)

    st.success(f"Prediction: **{label}**  |  Confidence: **{conf:.3f}**")
    st.dataframe(probs_table(probs), hide_index=True)


def ui_quiz(model: tf.keras.Model) -> None:
    st.header("Emotion Quiz")
    st.write(
        "Try to guess the emotion shown. The correct label is revealed after you submit."
    )

    items = build_quiz_items()

    if not items:
        st.error(
            "No quiz images found.\n\n"
            "Expected test dataset at: data_split/test/<class>/... \n"
            "Make sure the repository contains the test images and folder names match config.ACTIVE_CLASSES."
        )
        return

    # Session state init
    if "quiz_idx" not in st.session_state:
        st.session_state.quiz_idx = 0
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
    if "quiz_total" not in st.session_state:
        st.session_state.quiz_total = 0

    idx = int(st.session_state.quiz_idx) % len(items)
    item = items[idx]

    # Load image from path (lazy)
    img = Image.open(item.image_path).convert("RGB")
    st.image(img, use_container_width=True)

    guess = st.radio("Your guess:", cfg.ACTIVE_CLASSES, horizontal=True)

    cols = st.columns(3)
    with cols[0]:
        if st.button("Check"):
            st.session_state.quiz_total += 1
            if guess == item.label:
                st.session_state.quiz_score += 1
                st.success("Correct ✅")
            else:
                st.info(f"Correct label: **{item.label}**")

            # Optional: show model prediction (nice for demo)
            x = preprocess_pil(img)
            pred_label, pred_conf, _ = predict(model, x)
            st.caption(f"Model prediction: {pred_label} (confidence {pred_conf:.3f})")

    with cols[1]:
        if st.button("Next"):
            st.session_state.quiz_idx = idx + 1

    with cols[2]:
        if st.button("Reset"):
            st.session_state.quiz_idx = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_total = 0

    if st.session_state.quiz_total > 0:
        st.metric(
            "Quiz Score", f"{st.session_state.quiz_score}/{st.session_state.quiz_total}"
        )


# ============================================================
# Main
# ============================================================


def main() -> None:
    cfg.ensure_project_dirs()
    ui_header()
    mode = ui_sidebar()

    # Load model once; show user-friendly error if missing
    try:
        model = load_model()
    except Exception as e:
        st.error("Model could not be loaded.")
        st.code(str(e))
        st.stop()

    if mode == "Single Prediction":
        ui_single_prediction(model)
    elif mode == "Quiz Mode":
        ui_quiz(model)
    else:
        ui_about()


if __name__ == "__main__":
    main()
