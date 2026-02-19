# Installation Guide – Emotion Recognition System

## 1. System Requirements

- Windows 10/11
- Python 3.10 (recommended)
- PowerShell or compatible terminal
- Internet connection (for dependency installation)

---

## 2. Obtain the Project

Option A – Clone from GitHub:

```powershell
git clone https://github.com/DomDaBird/IU_Projekte.git
cd IU_Projekte/emotion-recognition
```

Option B – Extract from ZIP:

1. Download the ZIP archive.
2. Extract it.
3. Open PowerShell inside the `emotion-recognition` directory.

---

## 3. Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

---

## 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 5. Run the Application

```powershell
streamlit run app_streamlit.py
```

Open in browser:

http://localhost:8501

---

## 6. Optional: Train the Model

```powershell
python train.py
```

Model file will be saved to:

models/best_model.keras

---

## 7. Optional: Evaluate the Model

```powershell
python evaluate.py
```

Evaluation results will be stored in:

reports/eval/

---

## 8. Expected Folder Structure

The system expects:

data_split/
  train/
  val/
  test/

Each containing subfolders for:
- angry
- fear
- happy
- sad
- surprise

---

## 9. Troubleshooting

### TensorFlow not found

Ensure the virtual environment is activated before installing dependencies.

### Model file missing

Ensure `models/best_model.keras` exists in the models directory.

---

## 10. Configuration Management

All system parameters are defined in:

config.py

The repository is version-controlled and structured for maintainability and future extension.
