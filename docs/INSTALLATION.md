# Installation Guide (Windows)

This project was developed as part of IU course **DLBDSEAIS02 --
Artificial Intelligence** (Task 3). It provides a facial emotion
recognition system using transfer learning with EfficientNetB0.

## Requirements

-   Windows 10/11
-   Python 3.10+ (recommended)
-   (Optional) NVIDIA GPU + CUDA (training can also run on CPU, but
    slower)

## 1) Create and activate a virtual environment

Open PowerShell in the repository root:

``` bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

## 2) Install dependencies

``` bash
pip install -r requirements.txt
```

## 3) Verify the installation

Run a quick import check:

``` bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

If TensorFlow imports successfully, the environment is ready.

## Troubleshooting

If installation fails on TensorFlow:

``` bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
