# AI Facial Emotion Recognition Application via Streamlit

This project implements a deep learning–based facial emotion recognition system using transfer learning with EfficientNetB0.  
The application classifies facial images into five emotion categories:

- angry  
- fear  
- happy  
- sad  
- surprise  

The system includes:
- A Streamlit-based user interface (prediction + quiz mode)
- A training pipeline
- An evaluation pipeline
- Reproducible configuration and documentation

---

## Features

- Single image emotion prediction
- Interactive quiz mode
- Transfer learning with EfficientNetB0
- Confusion matrix and classification report
- Configuration-driven architecture
- Fully reproducible environment

---

## Quickstart (Demo Mode)

### 1️⃣ Clone repository

```powershell
git clone https://github.com/DomDaBird/IU_Projekte.git
cd IU_Projekte/emotion-recognition
```

### 2️⃣ Create virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Start the application

```powershell
streamlit run app_streamlit.py
```

Open in browser:
```
http://localhost:8501
```

---

## Training (Optional)

```powershell
python train.py
```

The best model will be saved in:

```
models/best_model.keras
```

---

## Evaluation (Optional)

```powershell
python evaluate.py
```

Evaluation outputs:
- reports/confusion_matrix.png
- reports/classification_report.txt
- reports/training_logs.json

---

## Repository Structure

```
emotion-recognition/
│
├─ app_streamlit.py
├─ train.py
├─ evaluate.py
├─ infer_single.py
├─ config.py
├─ model.py
├─ data.py
│
├─ models/
│   └─ best_model.keras
│
├─ reports/
│
├─ data_split/
│   ├─ train/
│   ├─ val/
│   └─ test/
│
├─ docs/
│   ├─ USER_GUIDE.md
│   └─ TECHNICAL_OVERVIEW.md
│
└─ requirements.txt
```

---

## Reproducibility & Maintainability

- Central configuration via config.py
- Dependencies defined in requirements.txt
- Version controlled using Git
- Clear separation of training, evaluation, and application logic
- Modular architecture for future extension

---

## Scenario Context

This system was developed in a simulated company scenario where a marketing department requested a deployable and documented emotion recognition application.  
The repository structure and documentation are designed to support future maintenance and extension within a configuration management system.
