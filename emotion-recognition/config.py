"""
Global configuration for the Emotion Recognition project.

This module centralizes:
- project paths
- class definitions
- model & data pipeline parameters
- training profiles (debug/dev/full)
- learning rate behavior

All other modules import this file to ensure consistent configuration.

Company scenario note:
Keeping all parameters here makes the system maintainable and reproducible
when checked into configuration management (e.g., Git).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent

DATA_DIR: Path = PROJECT_ROOT / "data_split"  # must contain train/val/test
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
CACHE_DIR: Path = PROJECT_ROOT / ".cache"

BEST_MODEL_NAME: str = "best_model.keras"

# ============================================================
# Reproducibility
# ============================================================

SEED: int = 42

# ============================================================
# Class definitions (order matters!)
# ============================================================

ACTIVE_CLASSES: List[str] = ["angry", "fear", "happy", "sad", "surprise"]

ALIASES: Dict[str, set[str]] = {
    "angry": {"angry", "anger"},
    "fear": {"fear", "scared"},
    "happy": {"happy", "happiness"},
    "sad": {"sad", "sadness"},
    "surprise": {"surprise", "surprised"},
}

# ============================================================
# Model parameters
# ============================================================

BackboneName = Literal["mobilenet_v2", "efficientnet_b0"]

BACKBONE: BackboneName = "efficientnet_b0"

# TensorFlow/Keras uses (H, W); PIL uses (W, H)
IMG_SIZE = (192, 192)
PIL_SIZE = (IMG_SIZE[1], IMG_SIZE[0])

# ============================================================
# Data pipeline
# ============================================================

BATCH_SIZE: int = 32
PREFETCH: int = 4
SHUFFLE_BUFFER_SIZE: int = 2048

# "roundrobin" = balanced training sampling, "none" = natural distribution
BALANCE_MODE: Literal["roundrobin", "none"] = "roundrobin"

# ============================================================
# Augmentation (if used in your pipeline)
# ============================================================

USE_AUG: bool = True
AUG_ROT: float = 0.08
AUG_ZOOM: float = 0.10
AUG_TRANS: float = 0.08
AUG_FLIP: bool = True
COLOR_JITTER: bool = True

# ============================================================
# Head + loss settings
# ============================================================

DROP_RATE: float = 0.25

USE_FOCAL: bool = True
FOCAL_GAMMA: float = 1.5

# ============================================================
# MixUp
# ============================================================

USE_MIXUP_STAGE1: bool = False
USE_MIXUP_STAGE2: bool = True
MIXUP_ALPHA: float = 0.1

LABEL_SMOOTH_STAGE1: float = 0.0
LABEL_SMOOTH_STAGE2: float = 0.0

# ============================================================
# Training profiles
# ============================================================

TrainMode = Literal["FAST_DEBUG", "DEV_TRAIN", "FULL_TRAIN"]


@dataclass(frozen=True)
class TrainProfile:
    epochs_stage1: int
    epochs_stage2: int
    train_fraction: float
    val_fraction: float
    lr_stage1: float
    lr_stage2: float
    finetune_layers: int
    earlystop1: int
    earlystop2: int


TRAIN_MODE: TrainMode = "FULL_TRAIN"

PROFILES: Dict[TrainMode, TrainProfile] = {
    "FAST_DEBUG": TrainProfile(
        epochs_stage1=2,
        epochs_stage2=3,
        train_fraction=0.20,
        val_fraction=0.50,
        lr_stage1=1e-3,
        lr_stage2=1e-5,
        finetune_layers=100,
        earlystop1=1,
        earlystop2=2,
    ),
    "DEV_TRAIN": TrainProfile(
        epochs_stage1=6,
        epochs_stage2=8,
        train_fraction=0.60,
        val_fraction=1.00,
        lr_stage1=1e-3,
        lr_stage2=1e-5,
        finetune_layers=140,
        earlystop1=2,
        earlystop2=4,
    ),
    "FULL_TRAIN": TrainProfile(
        epochs_stage1=8,
        epochs_stage2=15,
        train_fraction=1.00,
        val_fraction=1.00,
        lr_stage1=1e-3,
        lr_stage2=1e-4,
        finetune_layers=140,
        earlystop1=3,
        earlystop2=5,
    ),
}

_ACTIVE_PROFILE: TrainProfile = PROFILES[TRAIN_MODE]

EPOCHS_STAGE1: int = _ACTIVE_PROFILE.epochs_stage1
EPOCHS_STAGE2: int = _ACTIVE_PROFILE.epochs_stage2

LR_STAGE1: float = _ACTIVE_PROFILE.lr_stage1
LR_STAGE2: float = _ACTIVE_PROFILE.lr_stage2

FINETUNE_LAST_LAYERS: int = _ACTIVE_PROFILE.finetune_layers

TRAIN_FRACTION: float = _ACTIVE_PROFILE.train_fraction
VAL_FRACTION: float = _ACTIVE_PROFILE.val_fraction

EARLYSTOP_STAGE1_PATIENCE: int = _ACTIVE_PROFILE.earlystop1
EARLYSTOP_STAGE2_PATIENCE: int = _ACTIVE_PROFILE.earlystop2

FAST_TRAIN: bool = TRAIN_MODE != "FULL_TRAIN"

# ============================================================
# Learning rate schedule
# ============================================================

LrSchedule = Literal["cosine", "plateau"]

LR_SCHEDULE: LrSchedule = "cosine"

ROP_FACTOR: float = 0.5
ROP_PATIENCE: int = 2 if FAST_TRAIN else 3
ROP_MIN_LR: float = 1e-7
ROP_MIN_DELTA: float = 1e-4

# ============================================================
# Helpers (used by other scripts)
# ============================================================


def ensure_project_dirs() -> None:
    """Create required directories if they do not exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def validate_config() -> None:
    """
    Validate critical configuration values early.
    This helps future maintainers and prevents silent misconfiguration.
    """
    if BACKBONE not in ("mobilenet_v2", "efficientnet_b0"):
        raise ValueError(
            f"Invalid BACKBONE='{BACKBONE}'. Use 'mobilenet_v2' or 'efficientnet_b0'."
        )

    if not isinstance(IMG_SIZE, tuple) or len(IMG_SIZE) != 2:
        raise ValueError("IMG_SIZE must be a tuple like (H, W).")

    if len(ACTIVE_CLASSES) < 2:
        raise ValueError("ACTIVE_CLASSES must contain at least 2 classes.")

    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be > 0.")

    if LR_STAGE1 <= 0 or LR_STAGE2 <= 0:
        raise ValueError("Learning rates must be > 0.")

    if FINETUNE_LAST_LAYERS < 0:
        raise ValueError("FINETUNE_LAST_LAYERS must be >= 0.")


def set_global_seed(seed: int = SEED) -> None:
    """Set Python/NumPy/TensorFlow seeds for reproducibility."""
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


# Validate once on import (safe, fast)
validate_config()
