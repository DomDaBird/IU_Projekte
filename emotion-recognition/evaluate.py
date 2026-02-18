# evaluate.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config as cfg
from data import make_datasets


def load_best_model() -> tf.keras.Model:
    """
    Is used to load the best trained Keras model from disk.

    The path is determined from cfg.MODELS_DIR and cfg.BEST_MODEL_NAME.
    The model is loaded without recompiling, because it will only be used
    for inference during evaluation.
    """
    model_path = Path(cfg.MODELS_DIR) / cfg.BEST_MODEL_NAME
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    return model


def collect_predictions(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Is used to iterate once over the test dataset and collect predictions and labels.

    The function returns:
        y_true: ground truth labels as 1D integer array
        y_pred: predicted labels as 1D integer array
    """
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    for batch_x, batch_y in test_ds:
        probs = model.predict(batch_x, verbose=0)
        batch_pred = np.argmax(probs, axis=-1)

        if batch_y.shape[-1] == num_classes:
            batch_true = np.argmax(batch_y.numpy(), axis=-1)
        else:
            batch_true = batch_y.numpy().astype("int64")

        all_true.append(batch_true)
        all_pred.append(batch_pred)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    return y_true, y_pred


def save_reports(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    accuracy: float,
) -> None:
    """
    Is used to compute and store the textual classification report, the confusion
    matrix (CSV + PNG) and a small metrics JSON under cfg.REPORTS_DIR.
    """
    reports_dir = Path(cfg.REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ---- Classification report (text) ----
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=False,
    )
    (reports_dir / "classification_report.txt").write_text(report_str, encoding="utf-8")

    # ---- Confusion matrix (csv + png) ----
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    np.savetxt(reports_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ---- Metrics JSON ----
    report_dict: Dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
    )

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "test_accuracy": float(accuracy),
        "macro_avg": report_dict.get("macro avg", {}),
        "weighted_avg": report_dict.get("weighted avg", {}),
        "per_class": {c: report_dict.get(c, {}) for c in class_names},
    }
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )


def main() -> None:
    """
    Is used as the main entry point for the evaluation script.

    The following steps are executed:

    1. The data pipeline is instantiated to obtain the test dataset.
    2. The best saved model is loaded from disk.
    3. Predictions for the full test set are generated.
    4. Accuracy and a detailed sklearn classification report are printed.
    5. The report and confusion matrix are saved under cfg.REPORTS_DIR.
    """
    print("INFO | Starting evaluation script...")
    print(f"INFO | Data root: {cfg.DATA_DIR}")

    _, _, test_ds, class_names = make_datasets(cfg.DATA_DIR)
    num_classes = len(class_names)
    print(f"INFO | class_names (eval): {class_names}")

    model = load_best_model()
    print(f"INFO | Loaded model from: {Path(cfg.MODELS_DIR) / cfg.BEST_MODEL_NAME}")

    y_true, y_pred = collect_predictions(model, test_ds, num_classes)

    accuracy = float((y_true == y_pred).mean())
    print(f"\nINFO | Test accuracy = {accuracy:.4f}\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
        )
    )

    save_reports(y_true, y_pred, class_names, accuracy)
    print(f"✅ Results stored under: {cfg.REPORTS_DIR}")


if __name__ == "__main__":
    main()
