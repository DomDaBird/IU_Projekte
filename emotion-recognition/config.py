"""
This module contains all global configuration parameters for the project.

It centralizes paths, class names, model settings, augmentation settings,
training hyperparameters and learning rate behavior.

All other modules import this file to ensure consistent configuration.
"""

from __future__ import annotations
from pathlib import Path


# ============================================================
#  Project paths
# ============================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Main data directory (must contain train/val/test)
DATA_DIR: Path = PROJECT_ROOT / "data_split"

# Directory for trained model files
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Directory for evaluation reports (confusion matrices, metrics)
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# Temporary cache folder
CACHE_DIR: Path = PROJECT_ROOT / ".cache"

# Name of best model file saved during training
BEST_MODEL_NAME: str = "best_model.keras"


# ============================================================
#  Reproducibility settings
# ============================================================

SEED: int = 42


# ============================================================
#  Class definitions
# ============================================================

# Explicit list of active classes (order is important)
ACTIVE_CLASSES = ["angry", "fear", "happy", "sad", "surprise"]

# Aliases (optional dictionary to support matching external dataset labels)
ALIASES = {
    "angry": {"angry", "anger"},
    "fear": {"fear", "scared"},
    "happy": {"happy", "happiness"},
    "sad": {"sad", "sadness"},
    "surprise": {"surprise", "surprised"},
}


# ============================================================
#  Model parameters
# ============================================================

# Possible values: "mobilenet_v2", "efficientnet_b0"
BACKBONE: str = "efficientnet_b0"

# Image size used throughout the pipeline
IMG_SIZE = (192, 192)

# Image size convention:
# - TensorFlow / Keras datasets expect (H, W)
# - PIL expects (W, H)
PIL_SIZE = (IMG_SIZE[1], IMG_SIZE[0])

# ============================================================
#  Data pipeline parameters
# ============================================================

BATCH_SIZE = 32
PREFETCH = 4
SHUFFLE_BUFFER_SIZE = 2048

# "roundrobin" = balanced training batches
# "none" = natural class distribution
BALANCE_MODE = "roundrobin"


# ============================================================
#  Data augmentation settings
# ============================================================

USE_AUG = True

AUG_ROT = 0.08
AUG_ZOOM = 0.10
AUG_TRANS = 0.08
AUG_FLIP = True
COLOR_JITTER = True


# ============================================================
#  Head + Loss settings
# ============================================================

DROP_RATE = 0.25

# Whether focal loss is used instead of cross-entropy
USE_FOCAL = True
FOCAL_GAMMA = 1.5


# ============================================================
#  MixUp settings
# ============================================================

USE_MIXUP_STAGE1 = False
USE_MIXUP_STAGE2 = True
MIXUP_ALPHA = 0.1

LABEL_SMOOTH_STAGE1 = 0.0
LABEL_SMOOTH_STAGE2 = 0.0


# ============================================================
#  Training profiles (FAST_DEBUG / DEV_TRAIN / FULL_TRAIN)
# ============================================================

TRAIN_MODE = "FULL_TRAIN"

PROFILES = {
    "FAST_DEBUG": {
        "epochs_stage1": 2,
        "epochs_stage2": 3,
        "train_fraction": 0.20,
        "val_fraction": 0.50,
        "lr_stage1": 1e-3,
        "lr_stage2": 1e-5,
        "finetune_layers": 100,
        "earlystop1": 1,
        "earlystop2": 2,
    },
    "DEV_TRAIN": {
        "epochs_stage1": 6,
        "epochs_stage2": 8,
        "train_fraction": 0.60,
        "val_fraction": 1.00,
        "lr_stage1": 1e-3,
        "lr_stage2": 1e-5,
        "finetune_layers": 140,
        "earlystop1": 2,
        "earlystop2": 4,
    },
    "FULL_TRAIN": {
        "epochs_stage1": 8,
        "epochs_stage2": 15,
        "train_fraction": 1.00,
        "val_fraction": 1.00,
        "lr_stage1": 1e-3,
        "lr_stage2": 1e-4,
        "finetune_layers": 140,
        "earlystop1": 3,
        "earlystop2": 5,
    },
}

ACTIVE_PROFILE = PROFILES[TRAIN_MODE]

# Flatten parameters for easy import
EPOCHS_STAGE1 = ACTIVE_PROFILE["epochs_stage1"]
EPOCHS_STAGE2 = ACTIVE_PROFILE["epochs_stage2"]

LR_STAGE1 = ACTIVE_PROFILE["lr_stage1"]
LR_STAGE2 = ACTIVE_PROFILE["lr_stage2"]

FINETUNE_LAST_LAYERS = ACTIVE_PROFILE["finetune_layers"]

TRAIN_FRACTION = ACTIVE_PROFILE["train_fraction"]
VAL_FRACTION = ACTIVE_PROFILE["val_fraction"]

EARLYSTOP_STAGE1_PATIENCE = ACTIVE_PROFILE["earlystop1"]
EARLYSTOP_STAGE2_PATIENCE = ACTIVE_PROFILE["earlystop2"]

FAST_TRAIN = TRAIN_MODE != "FULL_TRAIN"


# ============================================================
#  Learning rate schedule
# ============================================================

# Possible values: "cosine", "plateau"
LR_SCHEDULE = "cosine"

ROP_FACTOR = 0.5
ROP_PATIENCE = 2 if FAST_TRAIN else 3
ROP_MIN_LR = 1e-7
ROP_MIN_DELTA = 1e-4
