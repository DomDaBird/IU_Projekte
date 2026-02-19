# User Guide – Emotion Recognition Application

## Purpose

This application predicts facial emotions from images using a trained deep learning model.  
Supported emotion classes:

- angry
- fear
- happy
- sad
- surprise

The system provides:
- Single image prediction
- Interactive quiz mode for demonstration

---

## Requirements

- Python 3.10 (recommended)
- Windows PowerShell or compatible terminal
- Internet connection for initial dependency installation

---

## Setup (Windows PowerShell)

Navigate to the project directory:

```powershell
cd emotion-recognition
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Important

Ensure that the trained model file exists:

```
models/best_model.keras
```

If the file is missing, the application will not start correctly.

---

## Run the Application

```powershell
streamlit run app_streamlit.py
```

Open in browser:

```
http://localhost:8501
```

---

## How to Use

### Single Image Prediction

1. Upload a facial image (JPG/PNG).
2. The predicted emotion and confidence score will be displayed.

### Quiz Mode

1. Switch to “Quiz Mode”.
2. Select the emotion that matches the displayed image.
3. The correct label is revealed after submission.

---

## Troubleshooting

### "No module named tensorflow"

Activate the virtual environment and reinstall dependencies:

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### "Model not found"

Verify that:

```
models/best_model.keras
```

exists in the project directory.

---

## Notes for Marketing Team

This application is intended as a demonstration tool.  
The trained model and test dataset are included in the repository to allow immediate execution without additional setup.
