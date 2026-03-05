# Evaluation Guide

The model is evaluated on a separate test dataset (`dataset/test/`).

## Run evaluation

``` bash
python evaluate.py
```

## Outputs

The evaluation should generate:

-   Accuracy
-   Precision / Recall / F1-score
-   Confusion matrix

Recommended output location:

-   reports/metrics.json
-   reports/classification_report.txt
-   reports/confusion_matrix.png
