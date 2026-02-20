from __future__ import annotations

"""
This module exposes a small helper function to predict the emotion for a single
image path.

It is intended as a reusable utility that can be imported from other scripts
(e.g. small demos or notebooks) without relying on the full Streamlit app.
The same preprocessing as in the main project is applied, and the trained
model is loaded from disk when needed.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

import config as cfg

# The path to the trained model is derived from the global project structure.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / cfg.BEST_MODEL_NAME

CLASS_NAMES: List[str] = cfg.ACTIVE_CLASSES
IMG_SIZE = cfg.IMG_SIZE


def load_model() -> tf.keras.Model:
    """
    The trained Keras model is loaded from disk.

    This function can be reused by small scripts that want to perform
    predictions outside the main training pipeline or Streamlit app.

    Returns
    -------
    model:
        Keras model ready for inference.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found at: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


def _preprocess_pil_image(img: Image.Image) -> np.ndarray:
    """
    A PIL image is converted into a NumPy batch suitable for model input.

    The image is converted to RGB, resized to the configured input shape,
    converted to float32, and expanded along the batch axis.

    Parameters
    ----------
    img:
        Input image as a PIL Image object.

    Returns
    -------
    arr:
        NumPy array with shape (1, H, W, 3) ready for inference.
    """
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(
    model: tf.keras.Model,
    image_path: Path,
) -> Tuple[str, float, np.ndarray]:
    """
    A single image on disk is classified into one of the emotion categories.

    Parameters
    ----------
    model:
        Keras model that is already loaded and ready for inference.
    image_path:
        Path to the image file that should be classified.

    Returns
    -------
    label:
        Predicted class label as a string.
    confidence:
        Probability value assigned to the predicted class.
    probs:
        Full probability vector over all classes as a NumPy array.
    """
    # The image is opened via PIL and preprocessed to the correct input shape.
    pil_img = Image.open(image_path)
    arr = _preprocess_pil_image(pil_img)

    # A prediction is performed and the first (and only) batch element is used.
    preds = model.predict(arr, verbose=0)
    probs = preds[0]

    # The index of the most probable class is found and mapped to a label.
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    confidence = float(probs[idx])

    return label, confidence, probs
