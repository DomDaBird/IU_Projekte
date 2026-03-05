# Training Guide

This project trains an EfficientNetB0-based classifier to detect facial
emotions.

## Dataset structure

The training pipeline expects a directory-based dataset:

dataset/ train/ angry/ fear/ happy/ sad/ surprise/ val/ angry/ fear/
happy/ sad/ surprise/ test/ angry/ fear/ happy/ sad/ surprise/

The full Kaggle dataset is not included in this repository.

## Recommended approach (Laptop-friendly)

For faster training on a laptop, a reduced dataset with balanced class
sizes is recommended (e.g., the same number of images per emotion).

## Start training

``` bash
python train.py
```

## Training outputs

Typical outputs include:

-   A saved model file (e.g., best_model.keras)
-   Training logs
-   Evaluation reports in `reports/`
