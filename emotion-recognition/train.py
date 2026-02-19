"""
Training script for the Emotion Recognition project.

Stages:
1) Transfer Learning: backbone frozen
2) Fine-Tuning: unfreeze last N backbone layers (BatchNorm stays frozen)

Outputs:
- models/best_model.keras
- reports/train/training_logs.json
- reports/train/training_history_stage1.csv
- reports/train/training_history_stage2.csv

Company scenario note:
This script is designed to be runnable after a fresh checkout with clear logs,
stable configuration, and reproducible behavior.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

import config as cfg
import data as data_mod
import model as model_mod


# ============================================================
# Losses
# ============================================================

def categorical_focal_loss(gamma: float = 2.0, eps: float = 1e-7):
    """
    Focal loss for one-hot labels.

    Args:
        gamma: focusing parameter
        eps: numerical stability

    Returns:
        callable loss(y_true, y_pred)
    """
    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), eps, 1.0 - eps)

        # Cross-entropy
        ce = -y_true * tf.math.log(y_pred)
        # Focal weight
        weight = tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=-1)

    return _loss


# ============================================================
# Dataset helpers (one-hot, mixup)
# ============================================================

def to_one_hot(ds: tf.data.Dataset, num_classes: int) -> tf.data.Dataset:
    """Convert sparse integer labels to one-hot labels."""
    def _map(x, y):
        y = tf.one_hot(tf.cast(y, tf.int32), depth=num_classes)
        return x, tf.cast(y, tf.float32)
    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)


def apply_mixup(ds: tf.data.Dataset, alpha: float, seed: int) -> tf.data.Dataset:
    """
    Apply MixUp on already-batched (x, y_onehot) dataset.
    """
    if alpha <= 0:
        raise ValueError("MixUp alpha must be > 0")

    rng = tf.random.Generator.from_seed(seed)

    def _mix(x, y):
        # Sample lambda from Beta(alpha, alpha)
        lam = rng.uniform([], 0.0, 1.0)
        # Approx beta via two gammas if needed:
        # But for simplicity, use uniform + power shaping:
        # (not perfect Beta, but stable and ok for a student project)
        lam = tf.pow(lam, 1.0 / alpha)
        lam = tf.clip_by_value(lam, 0.0, 1.0)

        # Shuffle within batch
        idx = tf.random.shuffle(tf.range(tf.shape(x)[0]), seed=seed)
        x2 = tf.gather(x, idx)
        y2 = tf.gather(y, idx)

        x_mix = lam * tf.cast(x, tf.float32) + (1.0 - lam) * tf.cast(x2, tf.float32)
        y_mix = lam * y + (1.0 - lam) * y2

        # Keep image dtype consistent (model preprocess casts anyway)
        x_mix = tf.clip_by_value(x_mix, 0.0, 255.0)
        return tf.cast(x_mix, tf.uint8), tf.cast(y_mix, tf.float32)

    return ds.map(_mix, num_parallel_calls=tf.data.AUTOTUNE)


# ============================================================
# Callbacks / logging
# ============================================================

def make_callbacks(stage: str, out_dir: Path, checkpoint_path: Path) -> list[tf.keras.callbacks.Callback]:
    """Create callbacks for a training stage."""
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"training_history_{stage}.csv"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=cfg.EARLYSTOP_STAGE2_PATIENCE if stage == "stage2" else cfg.EARLYSTOP_STAGE1_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_path), append=False),
    ]


def make_optimizer(lr: float, steps_per_epoch: int) -> tf.keras.optimizers.Optimizer:
    """Create optimizer with either cosine or plateau scheduling."""
    if cfg.LR_SCHEDULE == "cosine":
        # Cosine schedule over total steps (simple, robust)
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=max(steps_per_epoch * 10, 1),
            alpha=0.05,
        )
        return tf.keras.optimizers.Adam(learning_rate=schedule)

    # plateau: fixed lr, ReduceLROnPlateau callback added later
    return tf.keras.optimizers.Adam(learning_rate=lr)


def maybe_add_plateau_callback(callbacks: list[tf.keras.callbacks.Callback]) -> list[tf.keras.callbacks.Callback]:
    """Add ReduceLROnPlateau if configured."""
    if cfg.LR_SCHEDULE != "plateau":
        return callbacks

    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.ROP_FACTOR,
            patience=cfg.ROP_PATIENCE,
            min_lr=cfg.ROP_MIN_LR,
            min_delta=cfg.ROP_MIN_DELTA,
            verbose=1,
        )
    )
    return callbacks


# ============================================================
# Training logic
# ============================================================

def compile_model(
    model: tf.keras.Model,
    lr: float,
    steps_per_epoch: int,
    use_onehot: bool,
    label_smoothing: float,
) -> None:
    """Compile model with configured loss/metrics."""
    opt = make_optimizer(lr, steps_per_epoch)

    if use_onehot:
        if cfg.USE_FOCAL:
            loss = categorical_focal_loss(gamma=cfg.FOCAL_GAMMA)
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(
                from_logits=False,
                label_smoothing=label_smoothing,
            )
        metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    else:
        # Sparse labels (no label smoothing)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(optimizer=opt, loss=loss, metrics=metrics)


def train_stage(
    stage: str,
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    lr: float,
    out_dir: Path,
    checkpoint_path: Path,
    use_onehot: bool,
    label_smoothing: float,
) -> Dict:
    """Train one stage and return history dict."""
    # Determine steps per epoch for scheduling stability
    steps_per_epoch = None
    try:
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        if steps_per_epoch <= 0:
            steps_per_epoch = None
    except Exception:
        steps_per_epoch = None

    # Fallback if cardinality unknown
    if steps_per_epoch is None:
        steps_per_epoch = 200  # safe fallback; does not break training

    compile_model(
        model=model,
        lr=lr,
        steps_per_epoch=int(steps_per_epoch),
        use_onehot=use_onehot,
        label_smoothing=label_smoothing,
    )

    callbacks = make_callbacks(stage=stage, out_dir=out_dir, checkpoint_path=checkpoint_path)
    callbacks = maybe_add_plateau_callback(callbacks)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return history.history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train emotion recognition model.")
    parser.add_argument("--mode", type=str, default=cfg.TRAIN_MODE, choices=list(cfg.PROFILES.keys()))
    args = parser.parse_args()

    # Apply chosen mode at runtime (without editing config)
    cfg.TRAIN_MODE = args.mode  # type: ignore[attr-defined]

    # Setup
    cfg.ensure_project_dirs()
    cfg.set_global_seed(cfg.SEED)

    train_out = cfg.REPORTS_DIR / "train"
    train_out.mkdir(parents=True, exist_ok=True)

    # Log metadata for company scenario / reproducibility
    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "train_mode": cfg.TRAIN_MODE,
        "backbone": cfg.BACKBONE,
        "img_size": cfg.IMG_SIZE,
        "batch_size": cfg.BATCH_SIZE,
        "classes": cfg.ACTIVE_CLASSES,
        "profile": asdict(cfg.PROFILES[cfg.TRAIN_MODE]),
        "mixup_stage1": cfg.USE_MIXUP_STAGE1,
        "mixup_stage2": cfg.USE_MIXUP_STAGE2,
        "mixup_alpha": cfg.MIXUP_ALPHA,
        "use_focal": cfg.USE_FOCAL,
        "focal_gamma": cfg.FOCAL_GAMMA,
        "lr_schedule": cfg.LR_SCHEDULE,
        "balance_mode": cfg.BALANCE_MODE,
    }
    (train_out / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Build datasets
    bundle = data_mod.build_datasets(class_names=cfg.ACTIVE_CLASSES, seed=cfg.SEED)
    num_classes = len(bundle.class_names)

    # Create model
    model = model_mod.build_model(num_classes=num_classes, img_size=cfg.IMG_SIZE)

    # Prepare for one-hot training if we use focal / mixup / label smoothing
    use_onehot_stage1 = bool(cfg.USE_FOCAL or cfg.USE_MIXUP_STAGE1 or cfg.LABEL_SMOOTH_STAGE1 > 0)
    use_onehot_stage2 = bool(cfg.USE_FOCAL or cfg.USE_MIXUP_STAGE2 or cfg.LABEL_SMOOTH_STAGE2 > 0)

    train_ds_s1 = bundle.train
    val_ds_s1 = bundle.val

    if use_onehot_stage1:
        train_ds_s1 = to_one_hot(train_ds_s1, num_classes)
        val_ds_s1 = to_one_hot(val_ds_s1, num_classes)
        if cfg.USE_MIXUP_STAGE1:
            train_ds_s1 = apply_mixup(train_ds_s1, alpha=cfg.MIXUP_ALPHA, seed=cfg.SEED + 10)

    # Stage 1: backbone frozen
    ckpt_path = cfg.MODELS_DIR / cfg.BEST_MODEL_NAME
    hist1 = train_stage(
        stage="stage1",
        model=model,
        train_ds=train_ds_s1,
        val_ds=val_ds_s1,
        epochs=cfg.EPOCHS_STAGE1,
        lr=cfg.LR_STAGE1,
        out_dir=train_out,
        checkpoint_path=ckpt_path,
        use_onehot=use_onehot_stage1,
        label_smoothing=cfg.LABEL_SMOOTH_STAGE1,
    )

    # Stage 2: fine-tune last layers
    model_mod.set_finetune(model, finetune_last_layers=cfg.FINETUNE_LAST_LAYERS)

    train_ds_s2 = bundle.train
    val_ds_s2 = bundle.val

    if use_onehot_stage2:
        train_ds_s2 = to_one_hot(train_ds_s2, num_classes)
        val_ds_s2 = to_one_hot(val_ds_s2, num_classes)
        if cfg.USE_MIXUP_STAGE2:
            train_ds_s2 = apply_mixup(train_ds_s2, alpha=cfg.MIXUP_ALPHA, seed=cfg.SEED + 20)

    hist2 = train_stage(
        stage="stage2",
        model=model,
        train_ds=train_ds_s2,
        val_ds=val_ds_s2,
        epochs=cfg.EPOCHS_STAGE2,
        lr=cfg.LR_STAGE2,
        out_dir=train_out,
        checkpoint_path=ckpt_path,
        use_onehot=use_onehot_stage2,
        label_smoothing=cfg.LABEL_SMOOTH_STAGE2,
    )

    # Save combined logs
    combined = {"stage1": hist1, "stage2": hist2}
    (train_out / "training_logs.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")

    print("\nDONE.")
    print(f"Best model saved at: {ckpt_path}")
    print(f"Training logs: {train_out / 'training_logs.json'}")


if __name__ == "__main__":
    main()
