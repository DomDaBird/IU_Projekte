"""
Data loading and tf.data pipelines for the Emotion Recognition project.

This module provides:
- dataset discovery from folder structure (train/val/test)
- robust validation of class folders
- tf.data pipelines with deterministic behavior (seeded)
- optional balanced sampling (round-robin) for training

Expected structure:
data_split/
  train/<class>/*.jpg
  val/<class>/*.jpg
  test/<class>/*.jpg

Company scenario note:
Clear validation messages help non-developer stakeholders and future maintainers
run the system reliably after checkout from configuration management.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import tensorflow as tf

import config as cfg


SplitName = Literal["train", "val", "test"]


# ============================================================
# Data containers
# ============================================================

@dataclass(frozen=True)
class DatasetBundle:
    """Bundle for train/val/test datasets plus class names."""
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    class_names: List[str]


# ============================================================
# Utilities
# ============================================================

def _split_dir(split: SplitName) -> Path:
    return cfg.DATA_DIR / split


def _validate_data_layout(class_names: List[str]) -> None:
    """Validate that required directories exist and contain at least one image."""
    if not cfg.DATA_DIR.exists():
        raise FileNotFoundError(
            f"DATA_DIR not found: {cfg.DATA_DIR}\n"
            "Expected: emotion-recognition/data_split with train/val/test folders."
        )

    for split in ("train", "val", "test"):
        d = _split_dir(split)  # type: ignore[arg-type]
        if not d.exists():
            raise FileNotFoundError(f"Missing split folder: {d}")

        missing = [c for c in class_names if not (d / c).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing class folders in {d}: {missing}\n"
                "Class folder names must match config.ACTIVE_CLASSES."
            )

    # Quick sanity check: at least 1 file in each split
    for split in ("train", "val", "test"):
        d = _split_dir(split)  # type: ignore[arg-type]
        n = sum(1 for _ in d.rglob("*") if _.is_file())
        if n == 0:
            raise FileNotFoundError(f"No files found in split folder: {d}")


def _list_images_for_class(split: SplitName, class_name: str) -> List[Path]:
    """List image files for a given class in a split."""
    class_dir = _split_dir(split) / class_name
    # Common image extensions
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths)


def _apply_fraction(paths: List[Path], fraction: float, seed: int) -> List[Path]:
    """Deterministically subsample a list of paths by fraction."""
    if fraction >= 1.0:
        return paths
    if fraction <= 0.0:
        raise ValueError("Fraction must be > 0")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    k = max(1, int(len(paths) * fraction))
    idx = idx[:k]
    return [paths[i] for i in idx]


def describe_dataset(class_names: Optional[List[str]] = None) -> Dict[str, Dict[str, int]]:
    """
    Count files per split and per class.

    Returns:
        dict[split][class] -> count
    """
    if class_names is None:
        class_names = cfg.ACTIVE_CLASSES

    _validate_data_layout(class_names)

    out: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        out[split] = {}
        for c in class_names:
            out[split][c] = len(_list_images_for_class(split, c))  # type: ignore[arg-type]
    return out


# ============================================================
# tf.data building blocks
# ============================================================

def _decode_image(path: tf.Tensor) -> tf.Tensor:
    """Read and decode an image file to uint8 RGB tensor."""
    bytes_ = tf.io.read_file(path)
    img = tf.io.decode_image(bytes_, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return img


def _resize(img: tf.Tensor) -> tf.Tensor:
    """Resize to configured model input size (H, W)."""
    return tf.image.resize(img, cfg.IMG_SIZE, method="bilinear")


def _to_uint8(img: tf.Tensor) -> tf.Tensor:
    """
    Ensure uint8 range [0..255] for model preprocessing layer.
    """
    img = tf.clip_by_value(img, 0.0, 255.0)
    return tf.cast(img, tf.uint8)


def _augment(img: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    """Apply light augmentation (deterministic per-example when seeded)."""
    # Convert to float for augmentation ops
    x = tf.cast(img, tf.float32)

    if cfg.AUG_FLIP:
        x = tf.image.stateless_random_flip_left_right(x, seed=seed)

    # stateless random ops expect seed shape [2]
    if cfg.USE_AUG:
        # Random translation (approx) via random crop/pad
        if cfg.AUG_TRANS > 0:
            h = tf.shape(x)[0]
            w = tf.shape(x)[1]
            pad_h = tf.cast(tf.cast(h, tf.float32) * cfg.AUG_TRANS, tf.int32)
            pad_w = tf.cast(tf.cast(w, tf.float32) * cfg.AUG_TRANS, tf.int32)
            x = tf.image.resize_with_crop_or_pad(x, h + pad_h, w + pad_w)
            x = tf.image.stateless_random_crop(x, size=[h, w, 3], seed=seed)

        if cfg.AUG_ROT > 0:
            # small rotations: approximate via random 90-degree rotations is too coarse,
            # so we skip true rotation to avoid extra deps (tensorflow-addons)
            pass

        if cfg.AUG_ZOOM > 0:
            # Simple zoom: random central crop then resize back
            crop_frac = 1.0 - tf.minimum(tf.constant(cfg.AUG_ZOOM, tf.float32), 0.3)
            crop = tf.image.central_crop(x, crop_frac)
            x = tf.image.resize(crop, cfg.IMG_SIZE)

        if cfg.COLOR_JITTER:
            # Stateless brightness/contrast
            x = tf.image.stateless_random_brightness(x, max_delta=0.10, seed=seed)
            x = tf.image.stateless_random_contrast(x, lower=0.90, upper=1.10, seed=seed)

    x = tf.clip_by_value(x, 0.0, 255.0)
    return tf.cast(x, tf.uint8)


def _make_dataset_from_paths(
    paths: List[Path],
    labels: List[int],
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    """
    Build a tf.data dataset from lists of paths and integer labels.
    """
    path_ds = tf.data.Dataset.from_tensor_slices([str(p) for p in paths])
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    if training:
        ds = ds.shuffle(
            buffer_size=min(cfg.SHUFFLE_BUFFER_SIZE, max(len(paths), 1)),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def _map_fn(p: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        img = _decode_image(p)
        img = _resize(img)
        img = _to_uint8(img)

        if training and cfg.USE_AUG:
            # Create a stateless seed per element
            s = tf.random.experimental.stateless_fold_in(
                tf.constant([seed, seed + 1], dtype=tf.int32),
                tf.cast(tf.strings.to_hash_bucket_fast(p, 2**31 - 1), tf.int32),
            )
            img = _augment(img, s)

        return img, tf.cast(y, tf.int32)

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(cfg.PREFETCH)
    return ds


# ============================================================
# Public: dataset builders
# ============================================================

def build_datasets(
    class_names: Optional[List[str]] = None,
    seed: int = cfg.SEED,
) -> DatasetBundle:
    """
    Build train/val/test datasets according to config.

    - Training may use balanced sampling if cfg.BALANCE_MODE == "roundrobin"
    - Subsampling for fast/dev modes is supported via cfg.TRAIN_FRACTION / cfg.VAL_FRACTION

    Returns:
        DatasetBundle
    """
    if class_names is None:
        class_names = cfg.ACTIVE_CLASSES

    _validate_data_layout(class_names)

    # --------------------------
    # Collect paths per split/class
    # --------------------------
    train_per_class: Dict[str, List[Path]] = {c: _list_images_for_class("train", c) for c in class_names}
    val_per_class: Dict[str, List[Path]] = {c: _list_images_for_class("val", c) for c in class_names}
    test_per_class: Dict[str, List[Path]] = {c: _list_images_for_class("test", c) for c in class_names}

    # --------------------------
    # Apply fractions (dev/debug)
    # --------------------------
    if cfg.TRAIN_FRACTION < 1.0:
        train_per_class = {c: _apply_fraction(ps, cfg.TRAIN_FRACTION, seed + i) for i, (c, ps) in enumerate(train_per_class.items())}

    if cfg.VAL_FRACTION < 1.0:
        val_per_class = {c: _apply_fraction(ps, cfg.VAL_FRACTION, seed + 100 + i) for i, (c, ps) in enumerate(val_per_class.items())}

    # --------------------------
    # Build datasets
    # --------------------------
    if cfg.BALANCE_MODE == "roundrobin":
        train_ds = _build_balanced_train(train_per_class, class_names, seed)
    else:
        train_paths, train_labels = _flatten(train_per_class, class_names)
        train_ds = _make_dataset_from_paths(train_paths, train_labels, training=True, seed=seed)

    val_paths, val_labels = _flatten(val_per_class, class_names)
    test_paths, test_labels = _flatten(test_per_class, class_names)

    val_ds = _make_dataset_from_paths(val_paths, val_labels, training=False, seed=seed)
    test_ds = _make_dataset_from_paths(test_paths, test_labels, training=False, seed=seed)

    return DatasetBundle(train=train_ds, val=val_ds, test=test_ds, class_names=list(class_names))


def _flatten(per_class: Dict[str, List[Path]], class_names: List[str]) -> Tuple[List[Path], List[int]]:
    """Flatten per-class dict into (paths, labels) using class_names order."""
    paths: List[Path] = []
    labels: List[int] = []
    for idx, c in enumerate(class_names):
        ps = per_class.get(c, [])
        paths.extend(ps)
        labels.extend([idx] * len(ps))
    return paths, labels


def _build_balanced_train(
    train_per_class: Dict[str, List[Path]],
    class_names: List[str],
    seed: int,
) -> tf.data.Dataset:
    """
    Build a balanced training dataset using a round-robin strategy:
    each batch sees mixed classes more evenly.

    Implementation approach:
    - create a dataset per class
    - sample from datasets uniformly
    """
    # Build per-class datasets (unbatched initially)
    class_datasets = []
    for idx, c in enumerate(class_names):
        ps = train_per_class.get(c, [])
        if len(ps) == 0:
            continue

        labels = [idx] * len(ps)
        ds = tf.data.Dataset.from_tensor_slices(([str(p) for p in ps], labels))

        # Shuffle per class to avoid ordering artifacts
        ds = ds.shuffle(buffer_size=min(len(ps), cfg.SHUFFLE_BUFFER_SIZE), seed=seed + idx, reshuffle_each_iteration=True)

        def _map_fn(p: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            img = _decode_image(p)
            img = _resize(img)
            img = _to_uint8(img)

            if cfg.USE_AUG:
                s = tf.random.experimental.stateless_fold_in(
                    tf.constant([seed, seed + 1], dtype=tf.int32),
                    tf.cast(tf.strings.to_hash_bucket_fast(p, 2**31 - 1), tf.int32),
                )
                img = _augment(img, s)
            return img, tf.cast(y, tf.int32)

        ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        class_datasets.append(ds)

    if not class_datasets:
        raise RuntimeError("No training images found to build balanced dataset.")

    # Sample uniformly from class datasets
    ds = tf.data.Dataset.sample_from_datasets(
        class_datasets,
        weights=[1.0 / len(class_datasets)] * len(class_datasets),
        seed=seed,
    )

    # Batch + prefetch
    ds = ds.batch(cfg.BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(cfg.PREFETCH)
    return ds
