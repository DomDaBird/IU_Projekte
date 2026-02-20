"""
Model definition for the Emotion Recognition project.

This module provides:
- a serializable preprocessing layer (no Lambda)
- backbone selection (EfficientNetB0 / MobileNetV2)
- a compact, configurable classification head
- helper utilities for fine-tuning

Design goals (company handover):
- reproducible and maintainable
- safe saving/loading (Keras serialization)
- clear separation between model definition and training logic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import tensorflow as tf

import config as cfg

# ============================================================
# Serializable preprocessing (no Lambda)
# ============================================================


@tf.keras.utils.register_keras_serializable(package="emotion_recognition")
class BackbonePreprocess(tf.keras.layers.Layer):
    """
    Serializable preprocessing layer that applies the correct preprocessing
    function for the configured backbone.

    This avoids tf.keras.layers.Lambda (which can cause portability issues).
    """

    def __init__(self, backbone: str, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone

    def call(self, inputs, training=None):
        # Inputs expected in uint8/float32 range [0..255]
        x = tf.cast(inputs, tf.float32)

        if self.backbone == "efficientnet_b0":
            # EfficientNet preprocess expects [0..255] float images.
            return tf.keras.applications.efficientnet.preprocess_input(x)

        if self.backbone == "mobilenet_v2":
            # MobileNetV2 preprocess expects [0..255] float images.
            return tf.keras.applications.mobilenet_v2.preprocess_input(x)

        # Fallback: normalize to [0, 1]
        return x / 255.0

    def get_config(self):
        config = super().get_config()
        config.update({"backbone": self.backbone})
        return config


# ============================================================
# Head configuration
# ============================================================


@dataclass(frozen=True)
class HeadConfig:
    hidden_units: int = 256
    dropout: float = 0.25


# ============================================================
# Backbone builder
# ============================================================


def _build_backbone(
    img_size: Tuple[int, int],
    trainable: bool = False,
) -> tf.keras.Model:
    """
    Create the pretrained backbone with include_top=False.

    Args:
        img_size: (H, W)
        trainable: whether the entire backbone is trainable

    Returns:
        Keras model for feature extraction.
    """
    input_shape = (img_size[0], img_size[1], 3)

    if cfg.BACKBONE == "mobilenet_v2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    elif cfg.BACKBONE == "efficientnet_b0":
        base = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    else:
        raise ValueError(f"Invalid BACKBONE='{cfg.BACKBONE}'")

    base.trainable = trainable
    return base


# ============================================================
# Public API
# ============================================================


def build_model(
    num_classes: int,
    img_size: Tuple[int, int],
    head: Optional[HeadConfig] = None,
) -> tf.keras.Model:
    """
    Build the full classification model.

    Notes:
    - preprocessing is part of the model graph
    - backbone is frozen by default (transfer learning stage 1)
    - safe serialization (no Lambda layers)

    Args:
        num_classes: number of output classes
        img_size: (H, W)
        head: optional head configuration

    Returns:
        tf.keras.Model
    """
    if head is None:
        head = HeadConfig(hidden_units=256, dropout=cfg.DROP_RATE)

    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3), name="image")

    # Preprocessing in-graph ensures consistency across training & inference.
    x = BackbonePreprocess(cfg.BACKBONE, name="preprocess")(inputs)

    backbone = _build_backbone(img_size, trainable=False)
    x = backbone(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(head.dropout, name="drop1")(x)
    x = tf.keras.layers.Dense(head.hidden_units, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(head.dropout, name="drop2")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name=f"{cfg.BACKBONE}_fer{num_classes}"
    )
    return model


def set_finetune(
    model: tf.keras.Model,
    finetune_last_layers: int,
) -> None:
    """
    Unfreeze the last N layers of the backbone for fine-tuning.

    Args:
        model: full model returned by build_model()
        finetune_last_layers: number of backbone layers to unfreeze (>=0)
    """
    if finetune_last_layers < 0:
        raise ValueError("finetune_last_layers must be >= 0")

    # Find backbone (the first nested model that looks like an applications model)
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith(
            ("efficientnet", "mobilenet")
        ):
            backbone = layer
            break

    if backbone is None:
        raise RuntimeError("Backbone model not found inside the full model.")

    # Freeze all first
    backbone.trainable = True
    for l in backbone.layers:
        l.trainable = False

    # Unfreeze last N layers
    if finetune_last_layers == 0:
        return

    for l in backbone.layers[-finetune_last_layers:]:
        # Keep BatchNorm frozen for stability
        if isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = False
        else:
            l.trainable = True


def get_backbone_name() -> str:
    """Return configured backbone name."""
    return cfg.BACKBONE
