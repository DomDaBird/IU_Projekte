"""
Evaluation script for the Emotion Recognition project.

Evaluates the best saved model on the test dataset and writes:
- reports/eval/confusion_matrix.png
- reports/eval/confusion_matrix.csv
- reports/eval/classification_report.txt
- reports/eval/metrics.json

Company scenario note:
Outputs are generated as files so non-technical stakeholders can review
results, and future maintainers can reproduce and compare runs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config as cfg
import data as data_mod

# ============================================================
# Helpers
# ============================================================


def load_model(model_path: Path) -> tf.keras.Model:
    """Load a saved Keras model with custom layers."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Custom objects are auto-registered via @register_keras_serializable in model.py,
    # so compile=False is enough for inference/evaluation.
    return tf.keras.models.load_model(model_path, compile=False)


def ensure_eval_dir() -> Path:
    """Create evaluation output directory."""
    out = cfg.REPORTS_DIR / "eval"
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_predictions(
    model: tf.keras.Model, ds: tf.data.Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect y_true and y_pred for an integer-labeled dataset.

    Returns:
        y_true (N,), y_pred (N,)
    """
    y_true: List[int] = []
    y_pred: List[int] = []

    for x_batch, y_batch in ds:
        probs = model.predict(x_batch, verbose=0)
        pred = np.argmax(probs, axis=1).astype(int)

        y_true.extend(np.asarray(y_batch).astype(int).tolist())
        y_pred.extend(pred.tolist())

    return np.array(y_true, dtype=int), np.array(y_pred, dtype=int)


def save_confusion_matrix_png(
    cm: np.ndarray, class_names: List[str], out_path: Path
) -> None:
    """Save confusion matrix figure."""
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # annotate
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix_csv(
    cm: np.ndarray, class_names: List[str], out_path: Path
) -> None:
    """Save confusion matrix as CSV with labels."""
    header = ",".join(["true/pred"] + class_names)
    lines = [header]
    for i, row in enumerate(cm):
        lines.append(",".join([class_names[i]] + [str(int(v)) for v in row]))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# ============================================================
# Main
# ============================================================


def main() -> None:
    cfg.ensure_project_dirs()
    cfg.set_global_seed(cfg.SEED)

    eval_dir = ensure_eval_dir()

    model_path = cfg.MODELS_DIR / cfg.BEST_MODEL_NAME
    model = load_model(model_path)

    # Build datasets (we only need test, but bundle keeps class order consistent)
    bundle = data_mod.build_datasets(class_names=cfg.ACTIVE_CLASSES, seed=cfg.SEED)
    class_names = bundle.class_names

    # Predict
    y_true, y_pred = get_predictions(model, bundle.test)

    # Metrics
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    # Write outputs
    save_text(eval_dir / "classification_report.txt", report)

    save_confusion_matrix_png(cm, class_names, eval_dir / "confusion_matrix.png")
    save_confusion_matrix_csv(cm, class_names, eval_dir / "confusion_matrix.csv")

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(model_path),
        "classes": class_names,
        "num_samples": int(len(y_true)),
        "metrics": metrics,
    }
    (eval_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Console summary (for quick verification)
    print("\nEVALUATION DONE")
    print(f"Model: {model_path}")
    print(f"Samples: {len(y_true)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 macro:  {metrics['f1_macro']:.4f}")
    print(f"Saved:")
    print(f" - {eval_dir / 'confusion_matrix.png'}")
    print(f" - {eval_dir / 'confusion_matrix.csv'}")
    print(f" - {eval_dir / 'classification_report.txt'}")
    print(f" - {eval_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
