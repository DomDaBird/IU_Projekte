from __future__ import annotations

"""
This script provides a simple command line interface to run a single-image
emotion prediction with the trained model.

It loads the Keras model from disk, preprocesses the image from a given file
path, performs inference, and prints the predicted emotion together with
its confidence and the full probability distribution across classes.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

import config as cfg

# The location of the trained model is derived from the project configuration.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / cfg.BEST_MODEL_NAME

# The class order must match the order used during training.
CLASS_NAMES: List[str] = cfg.ACTIVE_CLASSES
IMG_SIZE = cfg.IMG_SIZE


def load_model() -> tf.keras.Model:
    """
    The trained Keras model is loaded from disk for inference.

    The model is loaded without compilation, because no further training is
    intended in this script. An error is raised if the model file does not
    exist at the expected location.

    Returns
    -------
    model:
        Keras model ready to be used for prediction.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found at: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


def _preprocess_image(path: Path) -> np.ndarray:
    """
    An image file is loaded from disk and converted into a model-ready array.

    The image is converted to RGB, resized to the input size specified in the
    configuration, and converted into a float32 NumPy array with a leading
    batch dimension.

    Parameters
    ----------
    path:
        Path to the input image file on disk.

    Returns
    -------
    arr:
        NumPy array with shape (1, H, W, 3) ready to be passed to the model.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)  # (width, height)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_single(
    model: tf.keras.Model,
    image_path: Path,
) -> Tuple[str, float, np.ndarray]:
    """
    A single image is classified with the loaded model.

    Parameters
    ----------
    model:
        Keras model used for inference.
    image_path:
        Path to the image file that should be classified.

    Returns
    -------
    label:
        Predicted class label as a string.
    confidence:
        Probability assigned to the predicted class.
    probs:
        Full probability vector over all classes as a NumPy array.
    """
    img_arr = _preprocess_image(image_path)

    # The model is asked to predict probabilities for the single image.
    preds = model.predict(img_arr, verbose=0)
    probs = preds[0]

    # The index of the most probable class is determined.
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    confidence = float(probs[idx])

    return label, confidence, probs


def _print_result(
    image_path: Path,
    label: str,
    confidence: float,
    probs: np.ndarray,
) -> None:
    """
    The prediction result is printed in a readable format to the console.

    The top-1 prediction with its confidence is shown first, followed by a
    tabular view of all class probabilities.
    """
    print(f"\nImage: {image_path}")
    print(f"Predicted emotion: {label} (confidence: {confidence:.3f})\n")

    print("Class probabilities:")
    for cls_name, p in zip(CLASS_NAMES, probs):
        print(f"  {cls_name:10s}: {p:.4f}")


def main() -> None:
    """
    The command line entry point is implemented.

    Usage:
        python predict_single.py path/to/image.jpg

    The model is loaded once, the given image is preprocessed, the prediction
    is performed, and the results are printed to standard output.
    """
    if len(sys.argv) < 2:
        print("Usage: python predict_single.py path/to/image.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    # The model is loaded once at the start of the script.
    model = load_model()

    # A prediction is performed for the provided image.
    label, confidence, probs = predict_single(model, image_path)

    # The result is printed in a human-readable form.
    _print_result(image_path, label, confidence, probs)


if __name__ == "__main__":
    main()
