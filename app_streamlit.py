"""
Streamlit application for the Emotion Recognition project.

Modes:
- Single Image Prediction
- Quiz Mode

This UI is designed for a company scenario where a marketing department
needs a simple and documented demo application.

Run:
  streamlit run app_streamlit.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

import config as cfg


# ============================================================
# Constants / Paths
# ============================================================

MODEL_PATH = cfg.MODELS_DIR / cfg.BEST_MODEL_NAME

# Quiz data sources:
# - preferred: dataset/test (real evaluation set)
# - fallback: demo_data (small sample shipped with the repo)
TEST_DIR = cfg.DATA_DIR / "test"
DEMO_DIR = cfg.PROJECT_ROOT / "demo_data"

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class QuizItem:
    """This dataclass stores a quiz image path and its ground-truth label."""
    image_path: Path
    label: str


# ============================================================
# Caching: model + quiz index
# ============================================================

@st.cache_resource
def load_model_cached() -> tf.keras.Model:
    """
    This function loads the trained model once per app session.

    The model is expected to include preprocessing in-graph (see model.py).
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file was not found.\n\n"
            f"Expected path: {MODEL_PATH}\n\n"
            "Next steps:\n"
            "1) Train the model: python train.py\n"
            f"2) Ensure the model is saved as: {cfg.BEST_MODEL_NAME} inside the 'models/' folder."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def build_quiz_items(source_dir: Path, limit_per_class: int, seed: int) -> List[QuizItem]:
    """
    This function builds a shuffled list of quiz items from a folder structure.

    Parameters
    ----------
    source_dir : Path
        Root directory containing emotion subfolders.
    limit_per_class : int
        Maximum images per class to index.
    seed : int
        Seed for deterministic shuffling.

    Returns
    -------
    List[QuizItem]
        List of quiz items containing (image_path, label).
    """
    rng = random.Random(seed)
    items: List[QuizItem] = []

    for label in cfg.ACTIVE_CLASSES:
        class_dir = source_dir / label
        if not class_dir.exists():
            continue

        paths = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
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
    This function converts a PIL image to a model input array.

    The model contains its own preprocessing layer, so this function only:
    - converts to RGB
    - resizes to cfg.PIL_SIZE
    - returns uint8 array in [0..255]
    """
    img = img.convert("RGB").resize(cfg.PIL_SIZE)
    arr = np.array(img, dtype=np.uint8)
    return np.expand_dims(arr, axis=0)


def validate_model_output(probs: np.ndarray) -> None:
    """This function validates that model output matches the configured class count."""
    if probs.ndim != 1:
        raise ValueError(f"Expected model output shape (C,), got {probs.shape}")
    if probs.shape[0] != len(cfg.ACTIVE_CLASSES):
        raise ValueError(
            "Model output size does not match cfg.ACTIVE_CLASSES.\n"
            f"Output size: {probs.shape[0]}\n"
            f"Expected: {len(cfg.ACTIVE_CLASSES)}"
        )


def predict(model: tf.keras.Model, img_arr: np.ndarray) -> Tuple[str, float, np.ndarray]:
    """This function returns (pred_label, confidence, probs)."""
    probs = model.predict(img_arr, verbose=0)[0]
    validate_model_output(probs)
    idx = int(np.argmax(probs))
    return cfg.ACTIVE_CLASSES[idx], float(probs[idx]), probs


def probs_table(probs: np.ndarray) -> List[Dict[str, object]]:
    """This function returns a table-friendly list for Streamlit."""
    return [{"class": c, "probability": float(p)} for c, p in zip(cfg.ACTIVE_CLASSES, probs)]


# ============================================================
# UI sections
# ============================================================

def ui_header() -> None:
    """This function renders the page header and basic configuration."""
    st.set_page_config(page_title="Emotion Recognition", page_icon="😊", layout="centered")
    st.title("Emotion Recognition Demo")
    st.caption("AI Facial Emotion Recognition Application via Streamlit")


def ui_sidebar() -> str:
    """This function renders the sidebar navigation and returns the selected mode."""
    st.sidebar.header("Navigation")
    mode = st.sidebar.radio("Mode", ["Single Prediction", "Quiz Mode", "About"], index=0)
    st.sidebar.markdown("---")
    st.sidebar.caption("Company scenario: built for a marketing demo and maintainable handover.")
    return mode


def ui_about() -> None:
    """This function renders the About page."""
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

    st.subheader("Data sources for quiz")
    st.write(
        "- Preferred: `dataset/test/<class>/...` (evaluation test set)\n"
        "- Fallback: `demo_data/<class>/...` (small demo set shipped with the repo)"
    )


def ui_single_prediction(model: tf.keras.Model) -> None:
    """This function renders the single image prediction page."""
    st.header("Single Image Prediction")
    st.write("Upload a facial image (JPG/PNG). The model returns the predicted emotion and probabilities.")

    file = st.file_uploader("Upload image", type=[e.replace(".", "") for e in VALID_EXTS])
    if not file:
        st.info("Upload an image to start.")
        return

    img = Image.open(file)
    st.image(img, use_container_width=True)

    x = preprocess_pil(img)
    label, conf, probs = predict(model, x)

    st.success(f"Prediction: **{label}**  |  Confidence: **{conf:.3f}**")
    st.dataframe(probs_table(probs), hide_index=True)


def _select_quiz_source() -> Tuple[Path, str]:
    """This function selects the quiz image source folder and returns (path, label)."""
    if TEST_DIR.exists():
        return TEST_DIR, "dataset/test"
    if DEMO_DIR.exists():
        return DEMO_DIR, "demo_data"
    return Path(), "none"


def ui_quiz(model: tf.keras.Model) -> None:
    """This function renders the quiz mode page."""
    st.header("Emotion Quiz")
    st.write("Try to guess the emotion shown. The correct label is revealed after you submit.")

    quiz_source, source_name = _select_quiz_source()
    if source_name == "none":
        st.error(
            "No quiz images found.\n\n"
            "Expected one of the following folders:\n"
            "- dataset/test/<class>/...\n"
            "- demo_data/<class>/...\n\n"
            "Folder names must match cfg.ACTIVE_CLASSES."
        )
        return

    if source_name == "dataset/test":
        st.caption("Quiz source: dataset/test (evaluation test set)")
        items = build_quiz_items(quiz_source, limit_per_class=300, seed=cfg.SEED)
    else:
        st.caption("Quiz source: demo_data (quick demo set)")
        items = build_quiz_items(quiz_source, limit_per_class=50, seed=cfg.SEED)

    if not items:
        st.error(
            f"No quiz images were indexed from: {quiz_source}\n\n"
            "Make sure each class folder contains image files (jpg/png/webp/...)."
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

            # Show model prediction (nice for a demo)
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
        st.metric("Quiz Score", f"{st.session_state.quiz_score}/{st.session_state.quiz_total}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    """This function runs the Streamlit application."""
    cfg.ensure_project_dirs()
    ui_header()
    mode = ui_sidebar()

    try:
        model = load_model_cached()
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