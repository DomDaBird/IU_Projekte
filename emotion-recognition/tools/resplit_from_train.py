# tools/resplit_from_train.py
"""
Utility script to create a train/val/test split from an existing train directory.

Current assumption:
    data_split/
        train/
            angry/
            fear/
            happy/
            sad/
            surprise/

After running this script, the structure will be:
    data_split/
        train/   # ~80% of images per class
        val/     # ~10% of images per class
        test/    # ~10% of images per class

Images are MOVED (not copied), so the original train-only layout is modified
in-place. The split is stratified by class and reproducible.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import shutil

import config as cfg


# ============================================================
#  Configuration
# ============================================================

# Base data directory from the project config, e.g. Path("data_split")
DATA_ROOT = Path(cfg.DATA_DIR)

# Fractions for the split (per class)
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

# Small epsilon so fractions do not need to sum to exactly 1.0
_EPS = 1e-6


# ============================================================
#  Helper functions
# ============================================================

def _assert_fractions() -> None:
    """
    Ensures that the chosen fractions are valid.
    """
    total = TRAIN_FRAC + VAL_FRAC + TEST_FRAC
    if abs(total - 1.0) > _EPS:
        raise ValueError(
            f"Split fractions must sum to 1.0, but got "
            f"{TRAIN_FRAC} + {VAL_FRAC} + {TEST_FRAC} = {total}"
        )


def _collect_class_files(train_dir: Path) -> Dict[str, List[Path]]:
    """
    Collects all image files per class from the existing train directory.

    Returns:
        A dict mapping class_name -> list of file Paths.
    """
    class_files: Dict[str, List[Path]] = {}

    for class_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        files = [p for p in class_dir.iterdir() if p.is_file()]
        class_files[class_dir.name] = files

    if not class_files:
        raise RuntimeError(f"No class subfolders found in {train_dir}")

    return class_files


def _prepare_target_dirs(data_root: Path, class_names: List[str]) -> Tuple[Path, Path, Path]:
    """
    Creates val/ and test/ directories with empty class subfolders.

    If val/ or test/ already exist and are non-empty, an error is raised to
    avoid accidentally reshuffling an already split dataset.
    """
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    for split_dir in [val_dir, test_dir]:
        if split_dir.exists():
            # check if directory is already populated
            non_empty = any(split_dir.iterdir())
            if non_empty:
                raise RuntimeError(
                    f"Target directory '{split_dir}' is not empty. "
                    "Please remove or rename existing val/test before running this script."
                )

    # Create split folders + class subfolders
    for split_dir in [val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
        for cname in class_names:
            (split_dir / cname).mkdir(parents=True, exist_ok=True)

    # Train directory and its class folders must already exist
    # (images will be moved out of these).
    return train_dir, val_dir, test_dir


def _split_indices(n: int) -> Tuple[int, int, int]:
    """
    Calculates how many samples go into train/val/test for a class of size n.
    The split is based on the global fractions, and rounding is handled so
    that the total stays equal to n.
    """
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    n_test = n - n_train - n_val  # remainder

    # Safety check
    if n_train < 0 or n_val < 0 or n_test < 0:
        raise RuntimeError(
            f"Negative split sizes computed for n={n}: "
            f"train={n_train}, val={n_val}, test={n_test}"
        )

    return n_train, n_val, n_test


def _move_files(
    files: List[Path],
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    class_name: str,
) -> None:
    """
    Moves the given file list into the corresponding split folders for
    a single class.
    """
    n = len(files)
    n_train, n_val, n_test = _split_indices(n)

    print(
        f"Class '{class_name}': total={n} -> "
        f"train={n_train}, val={n_val}, test={n_test}"
    )

    # Shuffle in-place, but deterministically using cfg.SEED
    rng = random.Random(cfg.SEED)
    rng.shuffle(files)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    assert len(train_files) == n_train
    assert len(val_files) == n_val
    assert len(test_files) == n_test

    for p in train_files:
        # stays in train/<class> (no move required)
        # The file is already in the correct place.
        continue

    for p in val_files:
        target = val_dir / class_name / p.name
        shutil.move(str(p), str(target))

    for p in test_files:
        target = test_dir / class_name / p.name
        shutil.move(str(p), str(target))


# ============================================================
#  Main entry point
# ============================================================

def main() -> None:
    """
    Creates a stratified train/val/test split from the current train directory.
    """
    _assert_fractions()

    data_root = DATA_ROOT
    train_dir = data_root / "train"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found at {train_dir}")

    print(f"INFO | Using train directory: {train_dir}")

    # Collect all class files from train/
    class_files = _collect_class_files(train_dir)
    class_names = sorted(class_files.keys())
    print(f"INFO | Detected classes: {class_names}")

    # Create val/ and test/ structure (train/ must already exist)
    train_dir, val_dir, test_dir = _prepare_target_dirs(data_root, class_names)
    print(f"INFO | Target dirs -> train={train_dir}, val={val_dir}, test={test_dir}")

    # Perform the split per class and move files
    for cname, files in class_files.items():
        _move_files(
            files=files,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            class_name=cname,
        )

    print("✅ Split completed successfully.")
    print("   Please re-run tools/check_data.py to verify the new distribution.")


if __name__ == "__main__":
    main()
