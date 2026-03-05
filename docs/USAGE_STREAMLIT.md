# Streamlit Demo Usage

The Streamlit app allows non-developers to upload an image and see the
predicted emotion with probabilities.

## 1) Start the app

From the repository root:

``` bash
streamlit run app_streamlit.py
```

## 2) Use demo images

This repository includes a small demo dataset:

`demo_data/`

You can test the app quickly by uploading images from the subfolders:

-   demo_data/angry/
-   demo_data/fear/
-   demo_data/happy/
-   demo_data/sad/
-   demo_data/surprise/

## 3) Model file requirement

The app expects a trained model file (e.g., `best_model.keras`).

If the model file is not available yet: - Train the model first (see
TRAINING.md), or - Add a trained model to the expected location.

## Expected output

-   Predicted emotion label
-   Probability distribution across the 5 classes
