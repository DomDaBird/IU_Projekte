import numpy as np
import tensorflow as tf

import config as cfg
from data import make_datasets


def check(ds: tf.data.Dataset, name: str, n_batches: int = 3) -> None:
    """
    A few batches from a dataset are inspected to verify label distribution.

    - Up to `n_batches` batches are iterated.
    - For each batch, the one-hot encoded labels are summed across the batch
      dimension to obtain a per-class count for that batch.
    - The summed vector is printed to give a quick impression of class balance.

    If the dataset is exhausted before `n_batches` batches are read, a message
    is printed and iteration stops.
    """
    print(f"\n== {name} ==")
    it = iter(ds)

    for i in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            print("...end of dataset (profiled subset).")
            break

        sums = np.sum(y.numpy(), axis=0)
        print(f"batch {i}: onehot_sum {sums}")


def main() -> None:
    """
    All three dataset splits are loaded and a small number of batches is inspected.

    - Datasets for train, validation and test are created by `make_datasets`.
    - Class names are printed.
    - The `check` helper is called for each split to print the distribution
      of one-hot labels in the first few batches.
    """
    train_ds, val_ds, test_ds, class_names = make_datasets(cfg.DATA_DIR)
    print("class_names:", class_names)

    check(train_ds, "TRAIN", n_batches=3)
    check(val_ds, "VAL", n_batches=1)
    check(test_ds, "TEST", n_batches=1)


if __name__ == "__main__":
    main()
