# pretrain_raf.py
"""
Utility script for pretraining the CNN backbone on the RAF-DB dataset.

The script assumes the following directory structure relative to the project root:

data_raf/
    train/
        angry/
        fear/
        happy/
        sad/
        surprise/
    val/
        ...
    test/
        ...

The same backbone architecture and configuration as in train.py are reused,
but the data source is switched from data_split/ to data_raf/.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import config as cfg

# ============================================================
# Paths and constants
# ============================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent
RAF_DIR: Path = PROJECT_ROOT / "data_raf"   # Expected RAF-DB root
RAF_TRAIN_DIR: Path = RAF_DIR / "train"
RAF_VAL_DIR: Path = RAF_DIR / "val"

CLASS_NAMES: List[str] = cfg.ACTIVE_CLASSES
IMG_SIZE = cfg.IMG_SIZE
BATCH_SIZE = cfg.BATCH_SIZE
SEED = cfg.SEED


# ============================================================
# Helper: dataset creation for RAF-DB
# ============================================================

def _raf_image_ds_from_dir(
    root: Path,
    subset_name: str,
) -> tf.data.Dataset:
    """
    Helper function used to create a Keras image dataset from a single directory.

    The directory is expected to contain one subdirectory per class (e.g. angry, fear, ...).
    Images are loaded as RGB and resized to the configured IMG_SIZE.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        directory=root,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True if subset_name == "train" else False,
        seed=SEED,
    )
    return ds


def make_raf_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    RAF-DB train and validation datasets are created and preprocessed.

    The configured class names from cfg.ACTIVE_CLASSES are used, so that
    the label order is kept consistent with the main FER pipeline.
    """
    if not RAF_TRAIN_DIR.exists():
        raise FileNotFoundError(f"RAF train directory not found at: {RAF_TRAIN_DIR}")
    if not RAF_VAL_DIR.exists():
        raise FileNotFoundError(f"RAF val directory not found at: {RAF_VAL_DIR}")

    train_ds = _raf_image_ds_from_dir(RAF_TRAIN_DIR, subset_name="train")
    val_ds = _raf_image_ds_from_dir(RAF_VAL_DIR, subset_name="val")

    # Basic performance optimizations are applied to the datasets.
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds, CLASS_NAMES


# ============================================================
# Class weighting helpers (same logic as in train.py)
# ============================================================

def raf_class_counts(train_root: Path, class_names: List[str]) -> np.ndarray:
    """
    Per-class image counts are computed by scanning the subdirectories.

    The order of counts follows the provided class_names list.
    """
    counts = []
    for cname in class_names:
        cdir = train_root / cname
        if not cdir.exists():
            counts.append(0.0)
            continue
        n = sum(1 for _ in cdir.iterdir() if _.is_file())
        counts.append(float(n))
    return np.asarray(counts, dtype=np.float32)


def compute_class_weights_raf(
    train_root: Path,
    class_names: List[str],
) -> Tuple[dict, np.ndarray]:
    """
    Class weights and a focal-loss alpha vector are derived from RAF-DB class counts.

    For class weights, the inverse of the class frequency is used and normalized.
    For the alpha vector (used in focal loss), the normalized weights are scaled
    so that the average alpha remains close to 1.0.
    """
    counts = raf_class_counts(train_root, class_names)

    # Inverse frequency weighting is applied.
    inv = 1.0 / np.maximum(counts, 1.0)
    inv = inv / inv.mean()

    class_w = {i: float(inv[i]) for i in range(len(class_names))}
    alpha_vec = inv / inv.sum() * len(inv)

    return class_w, alpha_vec.astype(np.float32)


# ============================================================
# Model construction (mirrors the main train.py logic)
# ============================================================

def build_backbone(input_shape) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    The feature-extractor backbone is constructed based on the configuration.

    Either MobileNetV2 or EfficientNetB0 is used, with ImageNet pretraining.
    The backbone model returns pooled feature vectors, not class logits.
    """
    if cfg.BACKBONE.lower() == "mobilenet_v2":
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            input_shape=input_shape,
            weights="imagenet",
        )
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    else:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            input_shape=input_shape,
            weights="imagenet",
        )
        preprocess = tf.keras.applications.efficientnet.preprocess_input

    inp = layers.Input(shape=input_shape)
    x = preprocess(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(cfg.DROP_RATE)(x)
    backbone = models.Model(inp, x, name="backbone")

    return backbone, base


def build_head(features: tf.Tensor, num_classes: int) -> tf.keras.layers.Layer:
    """
    The classification head is constructed on top of the backbone features.

    A single Dense layer with softmax is used to produce class probabilities.
    """
    return layers.Dense(
        num_classes,
        activation="softmax",
        name="cls_head",
    )(features)


def categorical_focal_loss(alpha_vec: np.ndarray, gamma: float):
    """
    A categorical focal loss function is created using the given alpha vector and gamma.

    The loss down-weights easy examples and focuses training on hard examples,
    while also rebalancing classes according to the alpha vector.
    """
    alpha = tf.constant(alpha_vec.astype("float32"), dtype=tf.float32)

    def loss(y_true, y_pred):
        # Predicted probabilities are clipped to avoid log(0).
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Standard categorical cross-entropy is computed.
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred_clipped), axis=-1)

        # pt is the predicted probability for the true class.
        pt = tf.reduce_sum(y_true * y_pred_clipped, axis=-1)

        # Per-class alpha weighting is applied.
        alpha_w = tf.reduce_sum(y_true * alpha, axis=-1)

        # Focal loss is computed according to the standard formula.
        fl = alpha_w * tf.pow(1.0 - pt, gamma) * ce
        return tf.reduce_mean(fl)

    return loss


def make_optimizer(total_steps: int, lr_base: float) -> tf.keras.optimizers.Optimizer:
    """
    An optimizer with an optional cosine learning-rate schedule is created.

    If LR_SCHEDULE is set to 'cosine' and total_steps is positive, a cosine decay
    schedule is used; otherwise, a constant learning rate is applied.
    """
    if cfg.LR_SCHEDULE == "cosine" and total_steps > 0:
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr_base,
            decay_steps=total_steps,
            alpha=0.1,
        )
        return tf.keras.optimizers.Adam(learning_rate=schedule)
    return tf.keras.optimizers.Adam(learning_rate=lr_base)


# ============================================================
# Training loop (single-stage or two-stage on RAF)
# ============================================================

def train_on_raf(
    epochs_backbone_frozen: int = 4,
    epochs_finetune: int = 6,
) -> None:
    """
    A two-stage training procedure on RAF-DB is executed.

    In stage 1, only the classification head is trained while the backbone is frozen.
    In stage 2, the last N layers of the backbone are unfrozen and fine-tuned.
    """
    # Datasets are created.
    train_ds, val_ds, class_names = make_raf_datasets()
    num_classes = len(class_names)

    # Class weights and focal alpha vector are computed.
    class_w, alpha_vec = compute_class_weights_raf(RAF_TRAIN_DIR, class_names)
    print(f"INFO | RAF class_weights = {class_w}")
    print(f"INFO | RAF alpha_vec(focal) = {alpha_vec}")

    # Model is constructed.
    input_shape = IMG_SIZE + (3,)
    backbone, base = build_backbone(input_shape)
    feats = backbone.output
    logits = build_head(feats, num_classes)
    model = tf.keras.Model(backbone.input, logits, name="raf_pretrain_model")

    # Stage 1: backbone is frozen.
    for layer in base.layers:
        layer.trainable = False

    train_batches = tf.data.experimental.cardinality(train_ds).numpy()
    total_steps_stage1 = int(train_batches * epochs_backbone_frozen)

    optimizer_stage1 = make_optimizer(total_steps_stage1, lr_base=cfg.LR_STAGE1)

    # Loss function selection is performed.
    if cfg.USE_FOCAL:
        loss_fn = categorical_focal_loss(alpha_vec=alpha_vec, gamma=cfg.FOCAL_GAMMA)
        class_weight = None
    else:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0)
        class_weight = class_w

    model.compile(
        optimizer=optimizer_stage1,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(PROJECT_ROOT / "models" / "raf_pretrain_stage1.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.EARLYSTOP_STAGE1_PATIENCE,
            restore_best_weights=True,
        ),
    ]

    print("INFO | RAF Stage 1: training head with backbone frozen...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_backbone_frozen,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Stage 2: fine-tuning is performed on the last FINETUNE_LAST_LAYERS layers.
    for layer in base.layers[:-cfg.FINETUNE_LAST_LAYERS]:
        layer.trainable = False
    for layer in base.layers[-cfg.FINETUNE_LAST_LAYERS:]:
        layer.trainable = True

    total_steps_stage2 = int(train_batches * epochs_finetune)
    optimizer_stage2 = make_optimizer(total_steps_stage2, lr_base=cfg.LR_STAGE2)

    model.compile(
        optimizer=optimizer_stage2,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc"),
        ],
    )

    callbacks_stage2 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(PROJECT_ROOT / "models" / "raf_pretrain_finetuned.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.EARLYSTOP_STAGE2_PATIENCE,
            restore_best_weights=True,
        ),
    ]

    print("INFO | RAF Stage 2: fine-tuning backbone...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_finetune,
        class_weight=class_weight,
        callbacks=callbacks_stage2,
        verbose=1,
    )

    print("✅ RAF pretraining finished. Models saved under models/.")


def main():
    """
    Entry point for RAF-DB pretraining.

    This function simply delegates to train_on_raf() with default epoch counts.
    """
    train_on_raf(
        epochs_backbone_frozen=4,
        epochs_finetune=6,
    )


if __name__ == "__main__":
    main()
