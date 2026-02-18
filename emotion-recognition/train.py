# train.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import os
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import config as cfg
from data import make_datasets
from mixup import apply_mixup

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
#  Helper functions for class statistics and weighting
# ============================================================

def class_counts_from_dir(train_root: Path, class_names: List[str]) -> Dict[str, float]:
    """
    Is used to count how many image files exist per class in a data directory.

    The function expects a structure like:
        train_root/
            angry/
            fear/
            happy/
            ...

    A separate subdirectory is expected for each class name.
    """
    counts: Dict[str, float] = {}
    for cname in class_names:
        class_dir = Path(train_root) / cname
        n = 0
        if class_dir.exists():
            for entry in os.scandir(class_dir):
                if entry.is_file():
                    n += 1
        counts[cname] = float(n)
    return counts


def effective_alpha(counts: np.ndarray, beta: float = 0.9999) -> np.ndarray:
    """
    Is used to compute class-balanced weights according to:
        "Class-Balanced Loss Based on Effective Number of Samples"
        (Cui et al., 2019).

    The returned vector can be used as alpha-weights for a focal loss.
    """
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / weights.sum() * len(counts)
    return weights.astype(np.float32)


# ============================================================
#  Model construction (backbone + classification head)
# ============================================================

def build_backbone(input_shape: Tuple[int, int, int]) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Is used to construct the CNN backbone with ImageNet weights.

    Depending on cfg.BACKBONE either MobileNetV2 or EfficientNetB0 is used.
    The base network is wrapped into a small model named "backbone" that
    already includes preprocessing, global average pooling and dropout.
    """
    backbone_name = cfg.BACKBONE.lower()

    if backbone_name == "mobilenet_v2":
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

    inp = layers.Input(shape=input_shape, name="image")
    x = preprocess(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(cfg.DROP_RATE, name="backbone_dropout")(x)

    backbone = models.Model(inp, x, name="backbone")
    return backbone, base


def build_head(features: tf.Tensor, num_classes: int) -> tf.Tensor:
    """
    Is used to add the final classification layer on top of the backbone.

    A dense layer with softmax activation is used, producing one probability
    value per class.
    """
    return layers.Dense(num_classes, activation="softmax", name="cls_head")(features)


# ============================================================
#  Loss and optimizer helpers
# ============================================================

def categorical_focal_loss(alpha_vec: np.ndarray, gamma: float):
    """
    Is used to construct a multi-class focal loss with class-dependent alpha.

    The provided alpha_vec is expected to contain one weight per class.
    """
    alpha = tf.constant(alpha_vec.astype("float32"), dtype=tf.float32)

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred_clipped), axis=-1)
        pt = tf.reduce_sum(y_true * y_pred_clipped, axis=-1)
        alpha_w = tf.reduce_sum(y_true * alpha, axis=-1)
        fl = alpha_w * tf.pow(1.0 - pt, gamma) * ce
        return tf.reduce_mean(fl)

    return loss


def make_optimizer(total_steps: int, lr_base: float) -> tf.keras.optimizers.Optimizer:
    """
    Is used to build the optimizer together with an optional learning-rate schedule.

    If cfg.LR_SCHEDULE is set to "cosine", a CosineDecay schedule is used.
    Otherwise, a constant learning rate is applied.
    """
    if cfg.LR_SCHEDULE == "cosine" and total_steps > 0:
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr_base,
            decay_steps=total_steps,
            alpha=0.1,
        )
        return tf.keras.optimizers.Adam(learning_rate=schedule)

    return tf.keras.optimizers.Adam(learning_rate=lr_base)


def compute_class_weights(
    train_root: Path,
    class_names: List[str],
) -> Tuple[Dict[int, float], np.ndarray]:
    """
    Is used to compute:

    * class_weight dict (for Keras fit, one weight per class index)
    * alpha_vec for focal loss (class-balanced weighting)

    Both are derived from the number of training samples per class.
    """
    counts_dict = class_counts_from_dir(train_root, class_names)
    print(f"INFO | counts(train)= {counts_dict}")

    counts_arr = np.array([counts_dict[c] for c in class_names], dtype=np.float32)
    total = counts_arr.sum()
    print("INFO | total_train_samples=", int(total))

    inv = total / np.maximum(counts_arr, 1.0)
    inv = inv / inv.mean()
    class_w = {i: float(inv[i]) for i in range(len(class_names))}

    alpha_vec = effective_alpha(counts_arr, beta=0.9999)
    return class_w, alpha_vec


# ============================================================
#  Stage-wise training (head-only + fine-tuning)
# ============================================================

def _find_base_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """
    Is used to robustly locate the actual CNN base model inside the full model.

    The base model is expected to be a child model of the "backbone" model,
    but common fallback names are also supported.
    """
    # Primary: "backbone" submodel
    try:
        backbone = model.get_layer("backbone")
        for l in backbone.layers:
            if isinstance(l, tf.keras.Model):
                return l
        for l in backbone.layers:
            if hasattr(l, "layers"):
                return l
    except Exception:
        pass

    # Fallback: any nested model with layers
    candidates = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    for m in candidates:
        if m.name.lower() in {"efficientnetb0", "mobilenetv2", "mobilenet_v2"}:
            return m

    raise ValueError(
        f"Backbone model not found. Available layers: {[l.name for l in model.layers]}"
    )


def train_one_stage(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    lr_base: float,
    class_weight: Dict[int, float],
    alpha_vec: np.ndarray,
    finetune_backbone: bool = False,
    unfreeze_last: int = 0,
    stage_name: str = "stage",
) -> tf.keras.callbacks.History:
    """
    Is used to train the model for one stage.

    In the first stage only the classification head is trained.
    In the second stage the last layers of the backbone are unfrozen and
    fine-tuned with a smaller learning rate.
    """
    if finetune_backbone and unfreeze_last > 0:
        base_model = _find_base_backbone(model)
        for layer in base_model.layers[-unfreeze_last:]:
            layer.trainable = True

    train_batches = tf.data.experimental.cardinality(train_ds).numpy()
    if train_batches < 0:
        train_batches = int(np.ceil(73017 / cfg.BATCH_SIZE))
    total_steps = int(train_batches * epochs)

    optimizer = make_optimizer(total_steps, lr_base)

    if cfg.USE_FOCAL:
        loss_fn = categorical_focal_loss(alpha_vec=alpha_vec, gamma=cfg.FOCAL_GAMMA)
        class_weight_arg = None
    else:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0)
        class_weight_arg = class_weight

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc"),
        ],
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(Path(cfg.MODELS_DIR) / cfg.BEST_MODEL_NAME),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
    )

    patience = cfg.EARLYSTOP_STAGE2_PATIENCE if finetune_backbone else cfg.EARLYSTOP_STAGE1_PATIENCE
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        restore_best_weights=True,
    )

    callbacks = [checkpoint_cb, earlystop_cb]

    if cfg.LR_SCHEDULE != "cosine":
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg.ROP_FACTOR,
                patience=cfg.ROP_PATIENCE,
                min_lr=cfg.ROP_MIN_LR,
                min_delta=cfg.ROP_MIN_DELTA,
            )
        )

    # CSV logger per stage (optional but handy)
    csv_log_path = Path(cfg.REPORTS_DIR) / f"training_logs_{stage_name}.csv"
    callbacks.append(tf.keras.callbacks.CSVLogger(str(csv_log_path), append=False))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight_arg,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def _save_training_logs(
    history_stage1: tf.keras.callbacks.History,
    history_stage2: tf.keras.callbacks.History | None,
    class_names: List[str],
    class_w: Dict[int, float],
    alpha_vec: np.ndarray,
) -> None:
    """
    Is used to store training logs and key configuration into reports/training_logs.json.
    """
    reports_dir = Path(cfg.REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "class_names": class_names,
        "class_weights": class_w,
        "alpha_vec_focal": alpha_vec.tolist(),
        "cfg_snapshot": {
            "BACKBONE": cfg.BACKBONE,
            "IMG_SIZE": cfg.IMG_SIZE,
            "BATCH_SIZE": cfg.BATCH_SIZE,
            "BALANCE_MODE": cfg.BALANCE_MODE,
            "USE_FOCAL": cfg.USE_FOCAL,
            "FOCAL_GAMMA": cfg.FOCAL_GAMMA,
            "USE_MIXUP_STAGE1": cfg.USE_MIXUP_STAGE1,
            "USE_MIXUP_STAGE2": cfg.USE_MIXUP_STAGE2,
            "MIXUP_ALPHA": cfg.MIXUP_ALPHA,
            "EPOCHS_STAGE1": cfg.EPOCHS_STAGE1,
            "EPOCHS_STAGE2": cfg.EPOCHS_STAGE2,
            "LR_STAGE1": cfg.LR_STAGE1,
            "LR_STAGE2": cfg.LR_STAGE2,
            "FINETUNE_LAST_LAYERS": cfg.FINETUNE_LAST_LAYERS,
            "LR_SCHEDULE": cfg.LR_SCHEDULE,
        },
        "stage1_history": history_stage1.history,
        "stage2_history": history_stage2.history if history_stage2 is not None else None,
    }

    (reports_dir / "training_logs.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8"
    )


# ============================================================
#  Main entry point
# ============================================================

def main() -> None:
    """
    Is used as the main entry point for the training script.

    The function performs the following high-level steps:

    1. Training/validation/test datasets are created.
    2. Class statistics and weights are computed.
    3. The backbone and classification head are built.
    4. Stage 1: only the head is trained (backbone frozen).
    5. Stage 2: the last backbone layers are fine-tuned.
    6. Training histories are saved to reports/training_logs.json.
    """
    train_ds, val_ds, _, class_names = make_datasets(cfg.DATA_DIR)
    num_classes = len(class_names)
    print(f"INFO | class_names={class_names} | num_classes={num_classes}")

    class_w_dict, alpha_vec = compute_class_weights(Path(cfg.DATA_DIR) / "train", class_names)
    print(f"INFO | class_weights= {class_w_dict}")
    print(f"INFO | alpha_vec(focal)= {alpha_vec}")

    train_stage1 = apply_mixup(train_ds, cfg.MIXUP_ALPHA) if cfg.USE_MIXUP_STAGE1 else train_ds
    train_stage2 = apply_mixup(train_ds, cfg.MIXUP_ALPHA) if cfg.USE_MIXUP_STAGE2 else train_ds

    input_shape = cfg.IMG_SIZE + (3,)
    backbone, base_model = build_backbone(input_shape)
    logits = build_head(backbone.output, num_classes)
    model = tf.keras.Model(backbone.input, logits, name="fer_model")

    for layer in base_model.layers:
        layer.trainable = False

    Path(cfg.MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(cfg.REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    print("INFO | Stage 1: training only the classification head (backbone frozen)")
    hist1 = train_one_stage(
        model=model,
        train_ds=train_stage1,
        val_ds=val_ds,
        epochs=cfg.EPOCHS_STAGE1,
        lr_base=cfg.LR_STAGE1,
        class_weight=class_w_dict,
        alpha_vec=alpha_vec,
        finetune_backbone=False,
        stage_name="stage1",
    )

    print(f"INFO | Stage 2: fine-tuning, last {cfg.FINETUNE_LAST_LAYERS} layers unfrozen")
    for layer in base_model.layers[:-cfg.FINETUNE_LAST_LAYERS]:
        layer.trainable = False
    for layer in base_model.layers[-cfg.FINETUNE_LAST_LAYERS:]:
        layer.trainable = True

    hist2 = train_one_stage(
        model=model,
        train_ds=train_stage2,
        val_ds=val_ds,
        epochs=cfg.EPOCHS_STAGE2,
        lr_base=cfg.LR_STAGE2,
        class_weight=class_w_dict,
        alpha_vec=alpha_vec,
        finetune_backbone=True,
        unfreeze_last=cfg.FINETUNE_LAST_LAYERS,
        stage_name="stage2",
    )

    _save_training_logs(hist1, hist2, class_names, class_w_dict, alpha_vec)

    print(f"✅ Finished. Models & logs are stored in: {cfg.MODELS_DIR} / {cfg.REPORTS_DIR}")


if __name__ == "__main__":
    main()
