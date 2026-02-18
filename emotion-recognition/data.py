from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import tensorflow as tf
import numpy as np
import config as cfg

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
#  Helper: Load images from directory into tf.data.Dataset
# ============================================================

def _image_ds_from_dir(
    root: Path,
    class_names: List[str],
    img_size: Tuple[int, int],
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """
    Is used to create an *unbatched* tf.data.Dataset from a directory that
    contains one subfolder per class.

    Expected folder format:
        root/
            angry/
            fear/
            happy/
            ...

    Arguments:
        root        : path to the directory containing class subfolders
        class_names : explicit list of class names in correct order
        img_size    : desired (H, W) size for image resizing
        shuffle     : whether the dataset should be shuffled
        seed        : random seed for reproducible shuffling

    Returns:
        A dataset of (image, one_hot_label) pairs, not batched.
    """
    return tf.keras.utils.image_dataset_from_directory(
        directory=str(root),
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=img_size,
        shuffle=shuffle,
        seed=seed,
        batch_size=None,  # unbatched; batching is done later
    )


# ============================================================
#  Balanced training through per-class exact sampling
# ============================================================

def _balanced_train_ds_exact(train_root: Path, class_names: List[str]) -> tf.data.Dataset:
    """
    Is used to create a balanced training dataset.

    Procedure:
        1. The full training set is loaded once from `train_root`
           as an unbatched dataset of (image, one_hot_label) pairs.
        2. For each class index, a filtered view of this dataset is created.
           Only samples where argmax(label) equals the class index are kept.
        3. All per-class datasets are combined with
           `tf.data.Dataset.sample_from_datasets` using equal weights.

    This ensures that:
        - majority classes (e.g. 'happy') do not dominate batches
        - each class contributes the same sampling probability
        - the underlying directory structure remains standard
          (`train/<class_name>/...`).
    """
    num_classes = len(class_names)

    # Step 1: load complete training dataset (unbatched)
    base_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(train_root),
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=cfg.IMG_SIZE,
        shuffle=True,
        seed=cfg.SEED,
        batch_size=None,  # unbatched
    )

    per_class_datasets: List[tf.data.Dataset] = []

    # Step 2: create filtered dataset for each class index
    for class_index in range(num_classes):
        def _predicate(x, y, ci=class_index):
            # predicate returns True only for samples of class `ci`
            return tf.equal(tf.argmax(y, axis=-1), ci)

        ds_ci = base_ds.filter(_predicate)
        per_class_datasets.append(ds_ci)

    # Step 3: sample equally from all per-class datasets
    balanced = tf.data.Dataset.sample_from_datasets(
        per_class_datasets,
        weights=[1.0 / num_classes] * num_classes,
        seed=cfg.SEED,
    )

    return balanced


# ============================================================
#  Train/Val/Test creation
# ============================================================

def make_datasets(data_root: Path):
    """
    Is used to create training, validation and test datasets.

    Folder structure must be:
        data_root/
            train/
                class1/
                class2/
                ...
            val/
                class1/
                class2/
                ...
            test/
                class1/
                class2/
                ...

    Balancing is applied only to the training dataset and controlled
    by `cfg.BALANCE_MODE`:

        - "roundrobin": uses `_balanced_train_ds_exact`
        - any other value: uses the standard unbalanced loader

    Returns:
        train_ds    : (possibly balanced) training dataset
        val_ds      : validation dataset
        test_ds     : test dataset
        class_names : sorted list of class names
    """
    train_dir = Path(data_root) / "train"
    val_dir = Path(data_root) / "val"
    test_dir = Path(data_root) / "test"

    # class names inferred from subfolders in training directory
    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])

    expected = list(cfg.ACTIVE_CLASSES)
    if set(class_names) != set(expected):
        raise ValueError(
            f"Class folders != cfg.ACTIVE_CLASSES\n"
            f"folders:   {class_names}\n"
            f"expected:  {expected}"
        )
    
    # Force exact order from config (source of truth)
    class_names = expected

    # --------------------------------------------
    # TRAIN
    # --------------------------------------------
    if cfg.BALANCE_MODE == "roundrobin":
        train_raw = _balanced_train_ds_exact(train_dir, class_names)
    else:
        train_raw = _image_ds_from_dir(
            root=train_dir,
            class_names=class_names,
            img_size=cfg.IMG_SIZE,
            shuffle=True,
            seed=cfg.SEED,
        )

    # global shuffle for additional randomness
    train_raw = train_raw.shuffle(
        buffer_size=cfg.SHUFFLE_BUFFER_SIZE,
        seed=cfg.SEED,
        reshuffle_each_iteration=True,
    )

    # --------------------------------------------
    # VAL + TEST (no balance, no shuffle)
    # --------------------------------------------
    val_raw = _image_ds_from_dir(
        root=val_dir,
        class_names=class_names,
        img_size=cfg.IMG_SIZE,
        shuffle=False,
        seed=cfg.SEED,
    )

    test_raw = _image_ds_from_dir(
        root=test_dir,
        class_names=class_names,
        img_size=cfg.IMG_SIZE,
        shuffle=False,
        seed=cfg.SEED,
    )

    # --------------------------------------------
    # PIPELINE: batching + prefetching
    # --------------------------------------------
    train_ds = (
        train_raw
        .batch(cfg.BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_raw
        .batch(cfg.BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        test_raw
        .batch(cfg.BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, class_names
