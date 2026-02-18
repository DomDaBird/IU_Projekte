"""
This module provides a simple MixUp augmentation function for tf.data pipelines.

MixUp blends two images and mixes their labels proportionally.
It is applied batch-wise and can be enabled independently for Stage 1 or Stage 2.
"""

from __future__ import annotations

import tensorflow as tf
import numpy as np


def _mixup_batch(
    images: tf.Tensor,
    labels: tf.Tensor,
    alpha: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Applies MixUp augmentation to a batch.

    Arguments:
        images : tensor of shape (B, H, W, C)
        labels : tensor of shape (B, num_classes)
        alpha  : strength of the Beta distribution for mixing

    Behavior:
        - A lambda value is sampled from Beta(alpha, alpha)
        - Each image is mixed with another randomly shuffled image
        - Labels are mixed using the same lambda

    Returns:
        mixed_images, mixed_labels
    """
    batch_size = tf.shape(images)[0]

    # Lambda for MixUp is sampled from a symmetric Beta distribution
    lam = np.random.beta(alpha, alpha)

    # A random permutation of the batch indices is applied
    index = tf.random.shuffle(tf.range(batch_size))

    # Mixed images
    mixed_x = lam * images + (1 - lam) * tf.gather(images, index)

    # Mixed labels
    mixed_y = lam * labels + (1 - lam) * tf.gather(labels, index)

    return mixed_x, mixed_y


def apply_mixup(dataset: tf.data.Dataset, alpha: float) -> tf.data.Dataset:
    """
    Wraps a dataset to apply MixUp to each batch.

    The dataset must emit (images, labels) pairs already batched.

    Returns:
        A transformed dataset with MixUp applied.
    """

    def _apply(images, labels):
        return _mixup_batch(images, labels, alpha=alpha)

    return dataset.map(_apply, num_parallel_calls=tf.data.AUTOTUNE)
