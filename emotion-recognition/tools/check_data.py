from __future__ import annotations
from pathlib import Path
from collections import Counter
import tensorflow as tf

from config import DATA_DIR, IMG_SIZE, BATCH_SIZE
from pathlib import Path as _P

TRAIN_DIR = _P(DATA_DIR) / "train"
VAL_DIR   = _P(DATA_DIR) / "val"
TEST_DIR  = _P(DATA_DIR) / "test"

def _count(ds: tf.data.Dataset) -> Counter:
    """
    Class frequencies are counted in a one-hot encoded dataset.

    The dataset is:
    - Unbatched to obtain individual samples.
    - Converted to NumPy via `as_numpy_iterator`.
    - The index of the maximum value in each one-hot label vector is used as
      the class id.

    A `Counter` mapping class indices to their counts is returned.
    """
    c = Counter()
    for _, y in ds.unbatch().as_numpy_iterator():
        c[int(y.argmax())] += 1
    return c

def main() -> None:
    """
    Basic sanity checks on the image directory structure are performed.

    Steps:
    1. The existence of the train and test directories is verified.
    2. Image datasets for train, optional validation, and test splits are
       created using `image_dataset_from_directory`.
    3. The inferred class names are printed.
    4. Per-class sample counts for train, validation (if available), and test
       splits are printed.
    """
    assert TRAIN_DIR.exists(), f"Missing: {TRAIN_DIR}"
    assert TEST_DIR.exists(), f"Missing: {TEST_DIR}"

    print(f"Train: {TRAIN_DIR}")
    print(f"Val:   {VAL_DIR} (exists={VAL_DIR.exists()})")
    print(f"Test:  {TEST_DIR}")

    train = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    class_names = train.class_names
    print("class_names:", class_names)

    val = None
    if VAL_DIR.exists() and any(VAL_DIR.iterdir()):
        val = tf.keras.utils.image_dataset_from_directory(
            VAL_DIR,
            label_mode="categorical",
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
        )

    test = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    print("Train counts:", _count(train))
    if val is not None:
        print("Val counts:", _count(val))
    print("Test counts:", _count(test))

if __name__ == "__main__":
    main()
