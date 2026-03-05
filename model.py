"""
Model definition for the Emotion Recognition project.

This module provides:
- a serializable preprocessing layer (no Lambda)
- backbone selection (EfficientNetB0 / MobileNetV2)
- a compact classification head
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
    This layer applies the correct preprocessing function for the configured backbone.

    The layer avoids tf.keras.layers.Lambda, which can cause portability issues when
    saving and loading models in different environments.
    """

    def __init__(self, backbone: cfg.BackboneName, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone

    def call(self, inputs, training=None):
        """
        This method applies preprocessing to input images.

        Inputs are expected in uint8/float32 range [0..255].
        """
        x = tf.cast(inputs, tf.float32)

        if self.backbone == "efficientnet_b0":
            return tf.keras.applications.efficientnet.preprocess_input(x)

        if self.backbone == "mobilenet_v2":
            return tf.keras.applications.mobilenet_v2.preprocess_input(x)

        # Fallback: normalize to [0, 1]
        return x / 255.0

    def get_config(self):
        """This method returns the layer configuration for Keras serialization."""
        config = super().get_config()
        config.update({"backbone": self.backbone})
        return config


# ============================================================
# Head configuration
# ============================================================

@dataclass(frozen=True)
class HeadConfig:
    """This dataclass stores configuration values for the classification head."""

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
    This function creates the pretrained backbone with include_top=False.

    Parameters
    ----------
    img_size : Tuple[int, int]
        Model input size (H, W).
    trainable : bool
        Whether the backbone should be trainable.

    Returns
    -------
    tf.keras.Model
        The backbone model used for feature extraction.
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
    This function builds the full classification model.

    Notes
    -----
    - Preprocessing is part of the model graph.
    - The backbone is frozen by default for transfer learning stage 1.
    - The model is safe to serialize because it avoids Lambda layers.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    img_size : Tuple[int, int]
        Model input size (H, W).
    head : Optional[HeadConfig]
        Optional head configuration.

    Returns
    -------
    tf.keras.Model
        The full Keras model (preprocess + backbone + head).
    """
    if head is None:
        head = HeadConfig(hidden_units=256, dropout=cfg.DROP_RATE)

    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3), name="image")

    # Preprocessing in-graph ensures consistent behaviour for training and inference.
    x = BackbonePreprocess(cfg.BACKBONE, name="preprocess")(inputs)

    backbone = _build_backbone(img_size, trainable=False)
    x = backbone(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(head.dropout, name="drop1")(x)
    x = tf.keras.layers.Dense(head.hidden_units, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(head.dropout, name="drop2")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"{cfg.BACKBONE}_fer_{num_classes}",
    )
    return model


def _find_backbone(model: tf.keras.Model) -> Optional[tf.keras.Model]:
    """
    This function tries to find the backbone model inside a full Keras model.

    It searches for the first nested tf.keras.Model that looks like an applications backbone.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            name = layer.name.lower()
            if "efficientnet" in name or "mobilenet" in name:
                return layer
    return None


def set_finetune(model: tf.keras.Model, finetune_last_layers: int) -> None:
    """
    This function unfreezes the last N layers of the backbone for fine-tuning.

    BatchNormalization layers remain frozen for stability.

    Parameters
    ----------
    model : tf.keras.Model
        Full model returned by build_model().
    finetune_last_layers : int
        Number of backbone layers to unfreeze (>= 0).
    """
    if finetune_last_layers < 0:
        raise ValueError("finetune_last_layers must be >= 0")

    backbone = _find_backbone(model)
    if backbone is None:
        raise RuntimeError("Backbone model not found inside the full model.")

    backbone.trainable = True

    # Freeze all layers first
    for layer in backbone.layers:
        layer.trainable = False

    if finetune_last_layers == 0:
        return

    # Unfreeze last N layers (except BatchNorm)
    for layer in backbone.layers[-finetune_last_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


def get_backbone_name() -> str:
    """This function returns the configured backbone name."""
    return cfg.BACKBONE