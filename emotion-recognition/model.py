from __future__ import annotations
import tensorflow as tf
import config as cfg

def _backbone(img_size, trainable=False):
    if cfg.BACKBONE == "mobilenet_v2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=(img_size[0], img_size[1], 3),
            include_top=False, weights="imagenet"
        )
    elif cfg.BACKBONE == "efficientnet_b0":
        base = tf.keras.applications.EfficientNetB0(
            input_shape=(img_size[0], img_size[1], 3),
            include_top=False, weights="imagenet"
        )
    else:
        raise ValueError(f"Unbekannter BACKBONE: {cfg.BACKBONE}")
    base.trainable = trainable
    return base

def build_model(num_classes: int, img_size):
    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
    x = inputs
    # Kein Lambda → sauberes Speichern/Laden
    base = _backbone(img_size, trainable=False)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(cfg.DROP_RATE)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(cfg.DROP_RATE)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=f"{cfg.BACKBONE}_fer5")
    return model
