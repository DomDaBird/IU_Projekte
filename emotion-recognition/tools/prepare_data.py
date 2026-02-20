from __future__ import annotations

import math
import random
import shutil
from pathlib import Path
from typing import Tuple

from PIL import Image
from preprocess_faces import process_split

import config as cfg

# A fixed seed is used to obtain reproducible splits.
random.seed(cfg.SEED)

# ------- Helper functions -------


def _norm(s: str) -> str:
    """
    A string is normalized for label comparison.

    - Leading and trailing whitespace is removed.
    - The string is converted to lowercase.
    - Hyphens, underscores and spaces are removed.

    The normalized string is returned.
    """
    return s.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _map_label(name: str) -> str | None:
    """
    A folder or label name is mapped to one of the active target classes.

    The name is normalized and then compared against all alias sets defined
    in `cfg.ALIASES`. If a match is found, the corresponding canonical class
    name is returned. If no mapping is possible, `None` is returned.
    """
    n = _norm(name)
    for target, aliases in cfg.ALIASES.items():
        alias_set = {_norm(a) for a in aliases}
        if n in alias_set or n == target:
            return target
    return None


def _copy_tree_selected(src: Path, dst: Path, keep: set[str]) -> None:
    """
    A subset of class folders is copied from `src` to `dst`.

    - Only subdirectories whose (mapped) class label is contained in `keep`
      are considered.
    - Image files in supported formats are copied into class folders named
      by the mapped target label.
    - The directory structure is flattened to `dst/<class_name>/file.ext`.
    """
    dst.mkdir(parents=True, exist_ok=True)

    for cls_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        tgt = _map_label(cls_dir.name)
        if tgt is None or tgt not in keep:
            continue

        out = dst / tgt
        out.mkdir(parents=True, exist_ok=True)

        for f in cls_dir.rglob("*"):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            shutil.copy2(f, out / f.name)


def _split_folder_stratified(
    src_cls_dir: Path,
    dst_train: Path,
    dst_val: Path,
    dst_test: Path,
    val_frac: float,
    test_frac: float,
) -> None:
    """
    A single class folder is split into train/validation/test subsets.

    - All image files within `src_cls_dir` are collected and randomly shuffled.
    - The number of validation and test samples is computed using the
      provided fractions.
    - Files are copied into `dst_train`, `dst_val` and `dst_test` according
      to the split proportions.
    """
    files = [p for p in src_cls_dir.iterdir() if p.is_file()]
    random.shuffle(files)

    n = len(files)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    test = files[:n_test]
    val = files[n_test : n_test + n_val]
    train = files[n_test + n_val :]

    for p in test:
        shutil.copy2(p, dst_test / p.name)
    for p in val:
        shutil.copy2(p, dst_val / p.name)
    for p in train:
        shutil.copy2(p, dst_train / p.name)


def _ensure_clean_dir(path: Path) -> None:
    """
    A directory is recreated in a clean state.

    - If the directory already exists, it is removed recursively.
    - A fresh, empty directory is then created.

    This is used to avoid mixing old and newly prepared data.
    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# ------- Pipelines per source dataset -------


def _prep_fer2013(raw_root: Path, temp_root: Path) -> None:
    """
    The FER2013 dataset is filtered and copied into a temporary structure.

    Expected layout:
        raw_root/{train,val,test}/{class}

    - Only classes listed in `cfg.ACTIVE_CLASSES` are kept.
    - Data is copied into `temp_root/fer2013/{train,val,test}/{class}`.
    """
    keep = set(cfg.ACTIVE_CLASSES)

    for split in ["train", "val", "test"]:
        src = raw_root / split
        if not src.exists():
            continue

        dst = temp_root / "fer2013" / split
        _ensure_clean_dir(dst)
        _copy_tree_selected(src, dst, keep)


def _prep_rafdb(raw_root: Path, temp_root: Path) -> None:
    """
    The RAF-DB dataset is filtered and copied into a temporary structure.

    Expected layout:
        raw_root/{train,test}/{class}

    - Only classes listed in `cfg.ACTIVE_CLASSES` are kept.
    - Data is copied into `temp_root/rafdb/{train,test}/{class}`.
    - A dedicated validation split is not present and will be created later
      during global merging if needed.
    """
    keep = set(cfg.ACTIVE_CLASSES)

    for split in ["train", "test"]:
        src = raw_root / split
        if not src.exists():
            continue

        dst = temp_root / "rafdb" / split
        _ensure_clean_dir(dst)
        _copy_tree_selected(src, dst, keep)


def _prep_hfe(raw_root: Path, temp_root: Path) -> None:
    """
    The HFE dataset is prepared and split into train/validation/test.

    Expected layout:
        raw_root/train/{class}

    - The complete training set is copied to `temp_root/hfe/train_full/{class}`.
    - For each class, 10% of the samples are moved to validation and
      10% to test using a stratified split.
    - The resulting splits are stored under:
        temp_root/hfe/{train,val,test}/{class}
    """
    keep = set(cfg.ACTIVE_CLASSES)
    src = raw_root / "train"
    if not src.exists():
        return

    tmp = temp_root / "hfe" / "train_full"
    _ensure_clean_dir(tmp)
    _copy_tree_selected(src, tmp, keep)

    # A 10% validation and 10% test split is created for each class.
    for cls in sorted(p.name for p in tmp.iterdir() if p.is_dir()):
        (temp_root / "hfe" / "train" / cls).mkdir(parents=True, exist_ok=True)
        (temp_root / "hfe" / "val" / cls).mkdir(parents=True, exist_ok=True)
        (temp_root / "hfe" / "test" / cls).mkdir(parents=True, exist_ok=True)

        _split_folder_stratified(
            tmp / cls,
            temp_root / "hfe" / "train" / cls,
            temp_root / "hfe" / "val" / cls,
            temp_root / "hfe" / "test" / cls,
            val_frac=0.10,
            test_frac=0.10,
        )


def _face_align_all(temp_root: Path, out_root: Path, size: Tuple[int, int]) -> None:
    """
    All temporary datasets are face-aligned, resized and merged into a common layout.

    The final structure is:
        out_root/{train,val,test}/{class}

    For each source dataset (`fer2013`, `rafdb`, `hfe`):

    - Every split (`train`, `val`, `test`) is passed through `process_split` to
      detect and crop faces and to resize images to `size`.
    - Cropped images are copied into the output splits, and the source dataset
      name is encoded in the filename prefix so that the origin remains traceable.
    """

    # Target splits are recreated in a clean state.
    _ensure_clean_dir(out_root / "train")
    _ensure_clean_dir(out_root / "val")
    _ensure_clean_dir(out_root / "test")

    def merge_split(src_split: Path, dst_split: Path) -> None:
        """
        A single split from one source dataset is aligned and merged.

        - If `src_split` does not exist, no action is taken.
        - Images are first processed into a temporary `*_aligned` directory.
        - The aligned images are then copied into `dst_split`, with filenames
          prefixed by the source dataset name.
        """
        if not src_split.exists():
            return

        # A temporary directory for aligned images is created next to the source split.
        tmp_aligned = src_split.parent / (src_split.name + "_aligned")

        # All images in the split are face-aligned and resized.
        process_split(src_split, tmp_aligned, size)

        # The aligned images are merged into the final split directory.
        for cls_dir in sorted(p for p in tmp_aligned.iterdir() if p.is_dir()):
            out_cls = dst_split / cls_dir.name
            out_cls.mkdir(parents=True, exist_ok=True)

            for f in cls_dir.iterdir():
                if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue

                # The source dataset name is encoded as a prefix in the filename.
                source_name = src_split.parent.parent.name
                target = out_cls / f"{source_name}__{f.name}"
                shutil.copy2(f, target)

    # All configured sources are merged into the unified dataset.
    for source in ["fer2013", "rafdb", "hfe"]:
        base = temp_root / source
        if not base.exists():
            continue

        merge_split(base / "train", out_root / "train")
        merge_split(base / "val", out_root / "val")
        merge_split(base / "test", out_root / "test")


def main() -> None:
    """
    The full data preparation pipeline is executed.

    Steps:
    1. A temporary working directory under `cfg.CACHE_DIR/prepare_tmp` is
       created (existing content is removed).
    2. All configured raw datasets (FER2013, RAF-DB, HFE) are preprocessed
       and copied into the temporary structure.
    3. All sources are face-aligned, resized and merged into the unified
       dataset under `cfg.PROCESSED_DIR`.
    4. The list of active class names is written to `models/class_names.json`
       for later reuse during inference.
    """
    raw = cfg.RAW_DIR
    tmp = cfg.CACHE_DIR / "prepare_tmp"

    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    print("INFO | Preprocess FER2013 …")
    _prep_fer2013(raw / "fer2013", tmp)

    print("INFO | Preprocess RAF-DB …")
    _prep_rafdb(raw / "rafdb", tmp)

    print("INFO | Preprocess HFE …")
    _prep_hfe(raw / "hfe", tmp)

    print("INFO | Face align & merge …")
    _face_align_all(tmp, cfg.PROCESSED_DIR, cfg.IMG_SIZE)

    # The class-name list is stored once in a fixed order for inference.
    class_names = cfg.ACTIVE_CLASSES
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.MODELS_DIR / "class_names.json").write_text(
        __import__("json").dumps(class_names, indent=2),
        encoding="utf-8",
    )

    print(f"✅ Done. Unified data is stored under: {cfg.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
