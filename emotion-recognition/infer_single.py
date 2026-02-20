"""
CLI inference tool for the Emotion Recognition project.

Usage (PowerShell):
  python infer_single.py --image path\\to\\image.jpg
  python infer_single.py --image path\\to\\image.jpg --topk 3
  python infer_single.py --image path\\to\\image.jpg --save_json reports\\infer_result.json

This script is intended as a developer tool (not required for the Streamlit demo).

Company scenario note:
A CLI is useful for automation/testing and helps future maintainers validate
the model outside the UI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

import config as cfg

# ============================================================
# Helpers
# ============================================================


def load_model(model_path: Path) -> tf.keras.Model:
    """Load a saved Keras model for inference."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def load_image(image_path: Path) -> Image.Image:
    """Load an image from disk."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    return img


def preprocess_pil(img: Image.Image) -> np.ndarray:
    """
    Convert PIL image to model input array.

    The model contains its own backbone-specific preprocessing layer,
    so we only:
    - resize
    - convert to uint8 [0..255]
    """
    img = img.resize(cfg.PIL_SIZE)
    arr = np.array(img, dtype=np.uint8)
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


def topk(probs: np.ndarray, class_names: List[str], k: int) -> List[Tuple[str, float]]:
    """Return top-k (class, probability) pairs."""
    k = max(1, min(k, len(class_names)))
    idx = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in idx]


def save_json(path: Path, payload: Dict) -> None:
    """Write JSON output to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single image inference (emotion recognition)."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to image file.")
    parser.add_argument(
        "--topk", type=int, default=5, help="Show top-k predictions (default: 5)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(cfg.MODELS_DIR / cfg.BEST_MODEL_NAME),
        help="Path to .keras model file.",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default="",
        help="Optional path to save predictions as JSON.",
    )
    args = parser.parse_args()

    cfg.ensure_project_dirs()

    image_path = Path(args.image)
    model_path = Path(args.model)

    model = load_model(model_path)
    img = load_image(image_path)
    x = preprocess_pil(img)

    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = cfg.ACTIVE_CLASSES[pred_idx]
    pred_conf = float(probs[pred_idx])

    top = topk(probs, cfg.ACTIVE_CLASSES, args.topk)

    print("\nINFERENCE RESULT")
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Prediction: {pred_label} (confidence={pred_conf:.4f})")
    print("\nTop-k:")
    for name, p in top:
        print(f" - {name:10s}: {p:.4f}")

    if args.save_json:
        payload = {
            "image": str(image_path),
            "model": str(model_path),
            "prediction": pred_label,
            "confidence": pred_conf,
            "topk": [{"class": n, "prob": p} for n, p in top],
            "classes": cfg.ACTIVE_CLASSES,
        }
        save_json(Path(args.save_json), payload)
        print(f"\nSaved JSON: {args.save_json}")


if __name__ == "__main__":
    main()
