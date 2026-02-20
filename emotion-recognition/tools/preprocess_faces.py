from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

# MTCNN is expensive to re-initialize – it is instantiated only once.
try:
    from mtcnn.mtcnn import MTCNN

    _DETECTOR = MTCNN()
except Exception as e:
    _DETECTOR = None
    print(f"WARN | MTCNN not available: {e} (fallback to center crop is used)")


def _center_square_crop(img: Image.Image) -> Image.Image:
    """
    A centered square crop is created from the input image.

    The shorter side length is used as the crop size and the crop is placed
    in the center of the original image. The cropped image is returned.
    """
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def detect_and_crop(img: Image.Image, margin: float = 0.2) -> Image.Image:
    """
    A face region is detected and cropped from the input image.

    - If an MTCNN detector is available, the most confident detected face
      is selected.
    - A bounding box is expanded by a relative margin around the face.
    - If detection fails or no detector is available, a centered square
      crop is used instead.

    The cropped face image is returned as a PIL Image.
    """
    if _DETECTOR is None:
        return _center_square_crop(img)

    arr = np.asarray(img.convert("RGB"))
    try:
        res = _DETECTOR.detect_faces(arr)
    except Exception:
        res = []

    if not res:
        return _center_square_crop(img)

    # The detection with the highest confidence is selected.
    box = max(res, key=lambda r: r.get("confidence", 0))["box"]  # x, y, w, h
    x, y, bw, bh = box

    # A padded bounding box around the detected face is computed.
    x0 = int(max(0, x - margin * bw))
    y0 = int(max(0, y - margin * bh))
    x1 = int(min(arr.shape[1], x + bw + margin * bw))
    y1 = int(min(arr.shape[0], y + bh + margin * bh))

    return Image.fromarray(arr[y0:y1, x0:x1])


def process_split(src_split: Path, dst_split: Path, img_size: Tuple[int, int]) -> None:
    """
    A complete split directory is processed and face-cropped images are written.

    - For each class subdirectory in `src_split`, a corresponding class directory
      is created under `dst_split`.
    - Each supported image file is opened, a face region is detected and cropped,
      resized to `img_size`, and saved into the target class directory.
    - If face detection or processing fails, the original image file is copied
      unchanged as a fallback.

    Existing files in the target location are left untouched.
    """
    dst_split.mkdir(parents=True, exist_ok=True)

    for cls_dir in sorted(p for p in src_split.iterdir() if p.is_dir()):
        out_cls = dst_split / cls_dir.name.lower()
        out_cls.mkdir(parents=True, exist_ok=True)

        for p in cls_dir.rglob("*"):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue

            out_p = out_cls / p.name
            if out_p.exists():
                # Existing outputs are reused to avoid duplicate work.
                continue

            try:
                img = Image.open(p).convert("RGB")
                face = detect_and_crop(img, margin=0.2).resize(img_size, Image.BILINEAR)
                face.save(out_p)
            except Exception:
                # If processing fails, the original image file is copied instead.
                shutil.copy2(p, out_p)
