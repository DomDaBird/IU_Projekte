"""
Training script for the Emotion Recognition project.

Stages:
1) Transfer learning (backbone frozen)
2) Fine-tuning (unfreeze last N backbone layers; BatchNorm stays frozen)

Outputs:
- models/best_model.keras (local artifact)
- reports/train/run_config.json
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
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import tensorflow as tf

import config as cfg
import data as data_mod
import model as model_mod

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# Losses
# ============================================================

def categorical_focal_loss(gamma: float = 2.0, eps: float = 1e-7):
    """
    This function creates a categorical focal loss for one-hot targets.

    Parameters
    ----------
    gamma : float
        Focusing parameter for hard examples.
    eps : float
        Numerical stability epsilon.

    Returns
    -------
    callable
        Loss function (y_true, y_pred) -> loss per sample.
    """

    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), eps, 1.0 - eps)

        ce = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=-1)

    return _loss


# ============================================================
# Dataset utilities
# ============================================================

def ensure_one_hot(ds: tf.data.Dataset, num_classes: int) -> tf.data.Dataset:
    """
    This function ensures that labels are one-hot encoded.

    If labels are already one-hot, they are cast to float32.
    If labels are sparse integers, they are converted to one-hot.
    """

    def _map(x, y):
        y = tf.convert_to_tensor(y)
        if y.shape.rank == 1:
            y_oh = tf.one_hot(tf.cast(y, tf.int32), depth=num_classes)
            return x, tf.cast(y_oh, tf.float32)
        return x, tf.cast(y, tf.float32)

    return ds.map(_map, num_parallel_calls=AUTOTUNE)


def apply_mixup(ds: tf.data.Dataset, alpha: float, seed: int) -> tf.data.Dataset:
    """
    This function applies MixUp on an already-batched (x, y_onehot) dataset.

    The implementation uses a Beta(alpha, alpha) distribution approximated via
    two Gamma samples (standard trick), which is stable without extra dependencies.
    """
    if alpha <= 0:
        raise ValueError("MixUp alpha must be > 0")

    rng = tf.random.Generator.from_seed(seed)

    def _mix(x, y):
        batch_size = tf.shape(x)[0]

        g1 = rng.gamma(shape=[], alpha=alpha, beta=1.0)
        g2 = rng.gamma(shape=[], alpha=alpha, beta=1.0)
        lam = g1 / (g1 + g2 + 1e-7)
        lam = tf.cast(lam, tf.float32)

        idx = tf.random.shuffle(tf.range(batch_size), seed=seed)
        x2 = tf.gather(x, idx)
        y2 = tf.gather(y, idx)

        x = tf.cast(x, tf.float32)
        x2 = tf.cast(x2, tf.float32)

        x_mix = lam * x + (1.0 - lam) * x2
        y_mix = lam * y + (1.0 - lam) * y2

        x_mix = tf.clip_by_value(x_mix, 0.0, 255.0)
        return tf.cast(x_mix, tf.uint8), tf.cast(y_mix, tf.float32)

    return ds.map(_mix, num_parallel_calls=AUTOTUNE)


def limit_dataset_fraction(ds: tf.data.Dataset, fraction: float) -> tf.data.Dataset:
    """
    This function limits a dataset to a fraction of its elements.

    If the dataset cardinality is unknown, the dataset is returned unchanged.
    """
    if fraction >= 1.0:
        return ds
    if fraction <= 0.0:
        raise ValueError("fraction must be in (0, 1].")

    try:
        n = tf.data.experimental.cardinality(ds).numpy()
        if n is None or n <= 0:
            return ds
        take_n = max(int(n * fraction), 1)
        return ds.take(take_n)
    except Exception:
        return ds


# ============================================================
# Callbacks / logging
# ============================================================

def make_callbacks(
    stage: str,
    out_dir: Path,
    checkpoint_path: Path,
) -> list[tf.keras.callbacks.Callback]:
    """This function creates callbacks for a training stage."""
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"training_history_{stage}.csv"
    callbacks: list[tf.keras.callbacks.Callback] = [
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
            patience=(
                cfg.EARLYSTOP_STAGE2_PATIENCE
                if stage == "stage2"
                else cfg.EARLYSTOP_STAGE1_PATIENCE
            ),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_path), append=False),
    ]

    if cfg.LR_SCHEDULE == "plateau":
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


def make_optimizer(lr: float, steps_per_epoch: int) -> tf.keras.optimizers.Optimizer:
    """This function creates an Adam optimizer with optional cosine scheduling."""
    if cfg.LR_SCHEDULE == "cosine":
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=max(steps_per_epoch * 10, 1),
            alpha=0.05,
        )
        return tf.keras.optimizers.Adam(learning_rate=schedule)

    return tf.keras.optimizers.Adam(learning_rate=lr)


# ============================================================
# Training logic
# ============================================================

def compile_model(
    model: tf.keras.Model,
    lr: float,
    steps_per_epoch: int,
    label_smoothing: float,
) -> None:
    """This function compiles the model with configured loss and metrics."""
    opt = make_optimizer(lr, steps_per_epoch)

    if cfg.USE_FOCAL:
        loss = categorical_focal_loss(gamma=cfg.FOCAL_GAMMA)
    else:
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=label_smoothing,
        )

    metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    model.compile(optimizer=opt, loss=loss, metrics=metrics)


def estimate_steps_per_epoch(ds: tf.data.Dataset, fallback: int = 200) -> int:
    """This function estimates steps per epoch from dataset cardinality."""
    try:
        steps = tf.data.experimental.cardinality(ds).numpy()
        if steps is None or steps <= 0:
            return fallback
        return int(steps)
    except Exception:
        return fallback


def train_stage(
    stage: str,
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    lr: float,
    out_dir: Path,
    checkpoint_path: Path,
    label_smoothing: float,
) -> Dict:
    """This function trains a single stage and returns the history dictionary."""
    steps_per_epoch = estimate_steps_per_epoch(train_ds)

    compile_model(
        model=model,
        lr=lr,
        steps_per_epoch=steps_per_epoch,
        label_smoothing=label_smoothing,
    )

    callbacks = make_callbacks(stage=stage, out_dir=out_dir, checkpoint_path=checkpoint_path)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return history.history


def build_run_metadata(mode: cfg.TrainMode) -> dict:
    """This function builds a run metadata dictionary for reproducibility."""
    profile = cfg.PROFILES[mode]
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "train_mode": mode,
        "backbone": cfg.BACKBONE,
        "img_size": cfg.IMG_SIZE,
        "batch_size": cfg.BATCH_SIZE,
        "classes": cfg.ACTIVE_CLASSES,
        "profile": {
            "epochs_stage1": profile.epochs_stage1,
            "epochs_stage2": profile.epochs_stage2,
            "train_fraction": profile.train_fraction,
            "val_fraction": profile.val_fraction,
            "lr_stage1": profile.lr_stage1,
            "lr_stage2": profile.lr_stage2,
            "finetune_layers": profile.finetune_layers,
            "earlystop1": profile.earlystop1,
            "earlystop2": profile.earlystop2,
        },
        "mixup_stage1": cfg.USE_MIXUP_STAGE1,
        "mixup_stage2": cfg.USE_MIXUP_STAGE2,
        "mixup_alpha": cfg.MIXUP_ALPHA,
        "use_focal": cfg.USE_FOCAL,
        "focal_gamma": cfg.FOCAL_GAMMA,
        "lr_schedule": cfg.LR_SCHEDULE,
        "balance_mode": cfg.BALANCE_MODE,
        "data_dir": str(cfg.DATA_DIR),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the emotion recognition model.")
    parser.add_argument(
        "--mode",
        type=str,
        default=cfg.TRAIN_MODE,
        choices=list(cfg.PROFILES.keys()),
        help="Training profile to use.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(cfg.DATA_DIR),
        help="Path to dataset root (contains train/val/test).",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default=str(cfg.MODELS_DIR / cfg.BEST_MODEL_NAME),
        help="Output path for the best model (.keras).",
    )
    args = parser.parse_args()

    mode: cfg.TrainMode = args.mode  # type: ignore[assignment]
    profile = cfg.PROFILES[mode]

    cfg.ensure_project_dirs()
    cfg.set_global_seed(cfg.SEED)

    data_root = Path(args.data_dir)

    # Build datasets (already batched)
    train_ds, val_ds, _test_ds, class_names = data_mod.make_datasets(data_root)
    num_classes = len(class_names)

    # Optional: laptop-friendly fractions per profile
    train_ds = limit_dataset_fraction(train_ds, profile.train_fraction)
    val_ds = limit_dataset_fraction(val_ds, profile.val_fraction)

    # Ensure one-hot (required for focal loss, label smoothing, mixup)
    train_ds = ensure_one_hot(train_ds, num_classes)
    val_ds = ensure_one_hot(val_ds, num_classes)

    # Output folders
    train_out = cfg.REPORTS_DIR / "train"
    train_out.mkdir(parents=True, exist_ok=True)

    # Save run metadata
    metadata = build_run_metadata(mode)
    (train_out / "run_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # Create model
    model = model_mod.build_model(num_classes=num_classes, img_size=cfg.IMG_SIZE)

    # Checkpoint path (local artifact)
    ckpt_path = Path(args.model_out)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------
    # Stage 1: backbone frozen
    # ------------------------------------------------
    train_s1 = train_ds
    if cfg.USE_MIXUP_STAGE1:
        train_s1 = apply_mixup(train_s1, alpha=cfg.MIXUP_ALPHA, seed=cfg.SEED + 10)

    hist1 = train_stage(
        stage="stage1",
        model=model,
        train_ds=train_s1,
        val_ds=val_ds,
        epochs=profile.epochs_stage1,
        lr=profile.lr_stage1,
        out_dir=train_out,
        checkpoint_path=ckpt_path,
        label_smoothing=cfg.LABEL_SMOOTH_STAGE1,
    )

    # ------------------------------------------------
    # Stage 2: fine-tune last layers
    # ------------------------------------------------
    model_mod.set_finetune(model, finetune_last_layers=profile.finetune_layers)

    train_s2 = train_ds
    if cfg.USE_MIXUP_STAGE2:
        train_s2 = apply_mixup(train_s2, alpha=cfg.MIXUP_ALPHA, seed=cfg.SEED + 20)

    hist2 = train_stage(
        stage="stage2",
        model=model,
        train_ds=train_s2,
        val_ds=val_ds,
        epochs=profile.epochs_stage2,
        lr=profile.lr_stage2,
        out_dir=train_out,
        checkpoint_path=ckpt_path,
        label_smoothing=cfg.LABEL_SMOOTH_STAGE2,
    )

    combined = {"stage1": hist1, "stage2": hist2}
    (train_out / "training_logs.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )

    print("\nDONE.")
    print(f"Best model saved at: {ckpt_path}")
    print(f"Training logs: {train_out / 'training_logs.json'}")


if __name__ == "__main__":
    main()