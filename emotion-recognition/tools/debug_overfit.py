from __future__ import annotations

"""
Simplified and robust overfitting test.

This version forces the model to overfit a small subset
of real images. All regular training augmentations and 
balancing mechanisms are bypassed.

The goal:
    - train accuracy should approach ~1.0
If not, something is broken in the pipeline or model.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

import config as cfg
from train import build_backbone, build_head


# =====================================================================
# Load a RAW training dataset (no balancing, no shuffling)
# =====================================================================

def _load_raw_train_ds(train_root: Path):
    """
    Loads the train/ directory WITHOUT balancing, WITHOUT sampling tricks,
    WITHOUT shuffle. Pure deterministic directory loading.
    """
    class_names = sorted([p.name for p in train_root.iterdir() if p.is_dir()])

    ds = tf.keras.utils.image_dataset_from_directory(
        train_root,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=cfg.IMG_SIZE,
        shuffle=False,         # absolutely no randomisation
    )

    return ds, class_names


# =====================================================================
# Collect small subset into numpy
# =====================================================================

def _collect_small_subset(ds: tf.data.Dataset, max_samples: int = 400):
    xs, ys = [], []
    collected = 0

    for batch_x, batch_y in ds:
        xs.append(batch_x.numpy())
        ys.append(batch_y.numpy())
        collected += batch_x.shape[0]

        if collected >= max_samples:
            break

    x_small = np.concatenate(xs, axis=0)[:max_samples]
    y_small = np.concatenate(ys, axis=0)[:max_samples]

    return x_small, y_small


# =====================================================================
# Build minimal trainable model (full backbone unfrozen)
# =====================================================================

def _build_simple_model(num_classes: int) -> tf.keras.Model:
    input_shape = cfg.IMG_SIZE + (3,)
    backbone, base_model = build_backbone(input_shape)

    # Unfreeze everything → full trainability
    for l in base_model.layers:
        l.trainable = True

    feats = backbone.output
    logits = build_head(feats, num_classes)
    model = tf.keras.Model(backbone.input, logits, name="debug_overfit_model")

    return model


# =====================================================================
# Main routine
# =====================================================================

def main():
    tf.random.set_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    train_root = Path(cfg.DATA_DIR) / "train"

    # Load raw dataset (no balance, no augmentation)
    train_ds, class_names = _load_raw_train_ds(train_root)

    # Collect 400 samples
    x_small, y_small = _collect_small_subset(train_ds, max_samples=400)
    print(f"Loaded subset: {x_small.shape}, {y_small.shape}")

    # Build simple trainable model
    num_classes = len(class_names)
    model = _build_simple_model(num_classes)

    # Compile simple optimizer/loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    print("Starting overfit test...\n")

    # Train
    history = model.fit(
        x_small,
        y_small,
        batch_size=32,
        epochs=10,
        verbose=2,
        shuffle=True,
    )

    train_acc_last = history.history["accuracy"][-1]
    print(f"\ntrain_acc_last: {train_acc_last:.4f}")

    if train_acc_last > 0.90:
        print("✅ Overfit test PASSED — model can memorize the data.")
    else:
        print("❌ Overfit test FAILED — check preprocessing or model setup!")


if __name__ == "__main__":
    main()
