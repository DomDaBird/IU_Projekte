from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, List, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image

import config as cfg

"""
This module is used to perform emotion recognition on a single image.

The trained Keras model is loaded from disk and cached so that it can be reused
across multiple CLI calls or by other modules (e.g. the Streamlit app).
"""

# Path to the best-performing model that is saved during training
MODEL_PATH = Path(cfg.MODELS_DIR) / cfg.BEST_MODEL_NAME

# Global singleton cache for the loaded model instance
_model: tf.keras.Model | None = None


def load_model_cached() -> tf.keras.Model:
    """
    The trained Keras model is loaded once from disk and stored in a global
    singleton cache so that repeated calls can reuse the same model instance
    without reloading it from disk.
    """
    global _model
    if _model is None:
        print(f"INFO | Lade Modell aus: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def _ensure_pil_image(source: Union[str, Path, Image.Image]) -> Image.Image:
    """
    A file path or existing PIL.Image object is converted to an RGB PIL.Image
    instance so that all downstream processing receives a consistent image type.
    """
    if isinstance(source, (str, Path)):
        img = load_img(source, color_mode="rgb")
    elif isinstance(source, Image.Image):
        img = source.convert("RGB")
    else:
        raise TypeError(f"Unsupported image source type: {type(source)}")
    return img


def preprocess_image(
    source: Union[str, Path, Image.Image],
    target_size: Tuple[int, int] | None = None
) -> np.ndarray:
    """
    The input image is loaded or normalized to a PIL RGB image, resized to the
    configured model input size and converted to a NumPy batch tensor.

    Returned shape:
        (1, H, W, 3) with dtype float32 and raw pixel values in [0, 255].

    The actual normalization (e.g. EfficientNet/MobileNet preprocessing) is
    expected to be handled inside the saved Keras model itself.
    """
    if target_size is None:
        target_size = cfg.PIL_SIZE

    img = _ensure_pil_image(source)
    img = img.resize(target_size)

    arr = img_to_array(img)  # (H, W, 3), float32, 0–255
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


def predict_image(
    source: Union[str, Path, Image.Image]
) -> Tuple[str, float, np.ndarray, List[str]]:
    """
    A forward pass through the trained model is performed for a single input
    image and the prediction results are returned.

    Returns:
        - predicted_label (str): the class name with the highest probability.
        - confidence (float): the probability associated with the predicted class.
        - probs (np.ndarray): full probability vector of shape (num_classes,).
        - class_names (List[str]): list of class names in index order.
    """
    model = load_model_cached()
    x = preprocess_image(source)
    preds = model.predict(x, verbose=0)[0]  # (num_classes,)

    class_names = cfg.ACTIVE_CLASSES
    idx = int(np.argmax(preds))
    label = class_names[idx] if idx < len(class_names) else str(idx)
    confidence = float(preds[idx])

    return label, confidence, preds, class_names


def cli():
    """
    A simple command line interface is provided so that a single image can be
    passed as an argument and its predicted emotion can be printed together
    with the top-K class probabilities.
    """
    parser = argparse.ArgumentParser(
        description="Emotion recognition for a single image using the FER model."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to an image file (JPG/PNG) whose emotion should be predicted.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top-K classes whose probabilities should be printed.",
    )

    args = parser.parse_args()
    image_path = Path(args.image_path)

    if not image_path.exists():
        print(f"ERROR | Image not found: {image_path}")
        return

    label, conf, probs, class_names = predict_image(image_path)

    print(f"✅ Prediction: {label} (confidence={conf:.3f})")
    print("Top-K probabilities:")

    # Classes are sorted in descending order of probability and the top-K
    # entries are printed together with their class names.
    idxs = np.argsort(probs)[::-1][: args.topk]
    for i in idxs:
        cname = class_names[i]
        p = probs[i]
        print(f"  - {cname:9s}: {p:.4f}")


if __name__ == "__main__":
    cli()

# Example usage from the project root:
# (fer) cd C:\Users\domvo\Desktop\emotion-recognition-fer2013
# python infer_single.py path\to\your\image.png
