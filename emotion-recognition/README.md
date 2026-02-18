# Emotion Recognition App with CNN (Mixed Kaggle Emotion Datasets)

This repository contains a complete pipeline for facial emotion recognition based on
transfer learning with a convolutional neural network (MobileNetV2 or EfficientNetB0).
The model is trained on a mixed dataset derived from the following sources
and evaluated on a clean **train/val/test** split.

## Datasets (Kaggle)

This project uses the following datasets from Kaggle:

1. **Human Face Emotions** – https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions/
2. **Facial Expression Dataset** – https://www.kaggle.com/datasets/aadityasinghal/facial-expression-dataset
3. **Natural Human Face Images for Emotion Recognition** – https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition
4. **Facial Emotions Recognition Dataset** – https://www.kaggle.com/datasets/nabilsherif/facial-emotions-recognition-dataset

**Notes**
- Please review each dataset’s license/terms on Kaggle.
- The repository only contains scripts and evaluation artifacts; raw images are not shipped with the repo.


In addition to the training and evaluation scripts, the project includes:

- A **single-image inference script** (`infer_single.py`)
- An interactive **Streamlit web app** (`app_streamlit.py`) with:
  - Upload-based prediction
  - A quiz game using images from the test set

The code and comments are written in English so that the project can be used directly
as a submission for an academic module or as a portfolio project.

---

## 1. Project structure

A typical layout of the project directory looks as follows:

```text
emotion-recognition/
│
├── app_streamlit.py        # Streamlit web app (single prediction + quiz)
├── infer_single.py         # Inference for a single image on the CLI
├── train.py                # Main training pipeline (two-stage fine-tuning)
├── evaluate.py             # Evaluation, classification report, confusion matrix
├── data.py                 # Dataset loading and input pipeline utilities
├── config.py               # Central configuration (paths, hyperparameters, profiles)
├── mixup.py                # MixUp (and optionally CutMix) batch-level augmentation
│
├── models/
│   └── best_model.keras    # Trained model (Keras SavedModel)
│
├── data_split/
│   ├── train/              # Training images (subfolders per class)
│   ├── val/                # Validation images
│   └── test/               # Test images (also used for the quiz)
│
├── reports/
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── training_logs.json
│
├── tools/
│   ├── resplit_from_train.py   # Helper script for creating train/val/test splits
│   ├── pretrain_raf.py         # Optional: pretraining on RAF-DB
│   └── ...                     # Other experimental or legacy scripts (optional)
│
├── requirements.txt
└── README.md

Only the core scripts (train.py, evaluate.py, infer_single.py, app_streamlit.py,
data.py, config.py, mixup.py) plus the model file and data_split/ are needed
to reproduce the main results and run the app. Everything inside tools/ is optional
and used for data preparation or additional experiments.


2. Data
2.1. Expected directory layout

The main training and evaluation pipeline expects the following layout:

data_split/
    train/
        angry/
        fear/
        happy/
        sad/
        surprise/
    val/
        angry/
        fear/
        happy/
        sad/
        surprise/
    test/
        angry/
        fear/
        happy/
        sad/
        surprise/


Each subdirectory contains images for one emotion class.

The five active classes are configured in config.py via ACTIVE_CLASSES.

2.2. Splitting from a source dataset

If the original data is available in a single train/ directory, a reproducible
split into train/, val/ and test/ can be created with the helper script
in tools/resplit_from_train.py (not shown here). The script typically:

Reads all images from data_mixed/train/<class>/.

Shuffles them with a fixed random seed (cfg.SEED).

Splits them into train/val/test according to configured fractions.

Copies or moves the images into data_split/train, data_split/val, data_split/test.

This step is done once. Afterwards, all training and evaluation scripts use data_split/.


3. Installation and environment
3.1. Python and virtual environment

A Python version compatible with TensorFlow 2.15 (e.g. Python 3.10 on Windows)
is recommended. A new virtual environment can be created and activated as follows:

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate


All required libraries are then installed via:

pip install -r requirements.txt


The requirements.txt file is prepared to work with TensorFlow 2.15 CPU on Windows
and includes all dependencies for training, evaluation and the Streamlit app.

4. Configuration

All global configuration is kept in config.py. Important settings include:

DATA_DIR – root folder for the data split (usually PROJECT_ROOT / "data_split").

MODELS_DIR – directory for saving trained models (models/).

REPORTS_DIR – directory for saving evaluation artifacts (reports/).

ACTIVE_CLASSES – list of emotion labels in the fixed order used everywhere.

BACKBONE – backbone architecture ("mobilenet_v2" or "efficientnet_b0").

IMG_SIZE – input image size (e.g. (192, 192)).

BATCH_SIZE, SHUFFLE_BUFFER_SIZE, PREFETCH – input pipeline settings.

USE_AUG, AUG_ROT, AUG_ZOOM, AUG_TRANS, AUG_FLIP, COLOR_JITTER – image-level augmentation.

USE_MIXUP_STAGE1, USE_MIXUP_STAGE2, MIXUP_ALPHA – batch-level MixUp configuration.

USE_FOCAL, FOCAL_GAMMA – focal loss configuration.

TRAIN_MODE – profile selection ("FAST_DEBUG", "DEV_TRAIN", "FULL_TRAIN"), which controls:

Number of epochs per stage.

Learning rates.

Fine-tuning depth.

Early stopping patience.

By switching TRAIN_MODE, the same code can be used for a quick debug run or
for a full overnight training run.

5. Training
5.1. Main training script

The main training pipeline is implemented in train.py. It performs:

Dataset creation via make_datasets() from data.py:

Balanced training pipeline with per-class sampling.

Validation and test datasets without balancing.

Computation of class weights based on the per-class image counts.

Model construction:

Backbone (MobileNetV2 or EfficientNetB0) with ImageNet weights.

Global average pooling and dropout.

Final Dense softmax classification head.

Two-stage training:

Stage 1: Only the head is trained (backbone is frozen).

Stage 2: The last N layers of the backbone are unfrozen and fine-tuned.

Optional MixUp data augmentation.

Optional focal loss with per-class alpha.

Callbacks:

ModelCheckpoint (best model saved to models/best_model.keras).

EarlyStopping.

Optional ReduceLROnPlateau or cosine decay learning-rate schedule.

Training is started from the project root with:

python -m train

The final model is stored as models/best_model.keras.

6. Evaluation

The evaluation script evaluate.py:

Loads data_split/test as a tf.data.Dataset.

Loads the best model from models/best_model.keras.

Computes:

Overall test accuracy.

Per-class precision, recall and F1-score using scikit-learn.

A confusion matrix.

Saves:

reports/classification_report.txt – textual classification report.

reports/confusion_matrix.png – confusion matrix plot.

Optionally additional logs (e.g. JSON with metrics).

Evaluation is started via:

python -m evaluate

7. Single-image inference (CLI)

The script infer_single.py is used to run inference on a single image from the command line.

Typical behavior:

The model is loaded once from models/best_model.keras.

An image path is passed as an argument (e.g. --image path/to/img.png).

The image is resized to the configured IMG_SIZE.

A prediction is made and the top-1 class and confidence are printed.

Optionally, the full probability vector is printed.

python infer_single.py --image path/to/face.png

8. Streamlit web app

The Streamlit app app_streamlit.py provides a simple web-based UI with two modes:

Single Prediction

An image is uploaded via the browser.

The original image is displayed.

The model predicts the emotion and shows:

Predicted label.

Confidence.

A table of probabilities for all classes.

Emotion Quiz Game

Images are loaded from data_split/test.

A random image is shown.

The user guesses the emotion (radio buttons).

The guess is compared to:

The ground-truth label (from the folder).

The model prediction.

A running score is maintained in the Streamlit session state.

Class probabilities are shown for each quiz image.

The model is cached with @st.cache_resource so that it is loaded only once.
Test images used for the quiz are cached with @st.cache_data and preprocessed in advance.

The app is started with:

streamlit run app_streamlit.py

After starting Streamlit, a browser window with the interface is opened.

9. Optional: RAF-DB pretraining

The script tools/pretrain_raf.py (or pretrain_raf.py in the root) is provided as an optional
utility for pretraining the backbone on RAF-DB before fine-tuning on the mixed FER dataset.

The script assumes a data_raf/ directory with train/, val/, test/ subfolders.

The same backbone architecture and loss configuration are reused.

A two-stage training (frozen backbone + fine-tuning) is performed.

Pretrained weights are stored separately (e.g. models/raf_pretrain_finetuned.keras).

This step is not required for the main training run, but can be used for additional experiments
or as a demonstration of transfer learning between different facial-expression datasets.

10. Reproducibility and profiles

The project is designed to be reproducible and configurable:

A fixed random seed (cfg.SEED) is used for dataset splitting and training.

Profiles (FAST_DEBUG, DEV_TRAIN, FULL_TRAIN) allow quick switching between:

Very short debug runs (for checking the pipeline).

Medium-length development runs.

Full-scale runs for final results (e.g. overnight training).

All hyperparameters, paths and switches are centralized in config.py so that
experiments can be described and reproduced easily for the written report.

11. Known limitations

The model assumes frontal or near-frontal faces with a single dominant expression.

Real-world performance may degrade on:

Multiple faces in the same image.

Strong occlusions or extreme lighting conditions.

Emotions outside the five configured classes.

The project does not include face detection inside the training/evaluation pipeline;
images are assumed to already contain cropped faces.

12. Credits

Backbone architectures: MobileNetV2 and EfficientNetB0 from tf.keras.applications.

App framework: Streamlit.

The project was implemented as part of a robotics/data science study module
to demonstrate practical end-to-end machine learning skills:
data processing, training, evaluation, deployment and documentation.