from __future__ import annotations

"""
This module defines custom loss functions used in the emotion recognition project.

Currently, a categorical focal loss is provided, which can be used as a drop-in
replacement for standard categorical cross-entropy in highly imbalanced
multi-class classification problems.
"""

from typing import Callable

import numpy as np
import tensorflow as tf


def categorical_focal_loss(
    alpha_vec: np.ndarray,
    gamma: float = 2.0,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    A categorical focal loss function is constructed for multi-class classification.

    The implementation follows the general idea of the focal loss proposed for
    object detection, extended to the multi-class (softmax) case. Class-specific
    weighting factors (alpha) are used to counteract class imbalance, while the
    focusing parameter gamma reduces the contribution of well-classified samples.

    The loss is defined as:

        FL = - sum_c alpha_c * (1 - p_t)^gamma * y_true_c * log(p_pred_c),

    where p_t is the predicted probability for the ground-truth class.

    Parameters
    ----------
    alpha_vec:
        One-dimensional NumPy array of shape (num_classes,) containing
        per-class weighting factors. Higher values increase the influence
        of the corresponding class in the overall loss.
    gamma:
        Focusing parameter. Larger values put more emphasis on hard
        misclassified examples and down-weight easy ones. A value of 0
        reduces the loss to weighted cross-entropy.

    Returns
    -------
    loss_fn:
        A Keras-compatible loss function that expects one-hot encoded
        `y_true` tensors and probability predictions `y_pred` from a
        softmax output layer.
    """
    # The alpha vector is converted into a constant tensor once to avoid
    # re-creating it on every loss call.
    alpha = tf.constant(alpha_vec.astype("float32"), dtype=tf.float32)

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        The focal loss for a batch of predictions is computed.

        Parameters
        ----------
        y_true:
            One-hot encoded labels with shape (batch, num_classes).
        y_pred:
            Predicted class probabilities with shape (batch, num_classes).

        Returns
        -------
        loss:
            Scalar tensor representing the mean focal loss over the batch.
        """
        # The predictions are clipped to avoid numerical issues with log(0).
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Standard categorical cross-entropy per sample is computed.
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred_clipped), axis=-1)

        # p_t corresponds to the probability of the ground-truth class.
        pt = tf.reduce_sum(y_true * y_pred_clipped, axis=-1)

        # Class-specific alpha weights are gathered according to y_true.
        alpha_weights = tf.reduce_sum(y_true * alpha, axis=-1)

        # The focal term (1 - p_t)^gamma is applied.
        focal_term = tf.pow(1.0 - pt, gamma)

        # The final loss per sample is constructed.
        loss_per_sample = alpha_weights * focal_term * ce

        # The mean over the batch is returned as the final loss value.
        return tf.reduce_mean(loss_per_sample)

    return loss_fn
