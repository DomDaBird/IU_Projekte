# Technical Overview – Emotion Recognition System

## System Architecture

Input Image  
→ Preprocessing (resize, normalization)  
→ CNN Model (EfficientNetB0)  
→ Softmax probabilities  
→ Predicted emotion label  

Optional:
→ Evaluation pipeline (confusion matrix, classification report)

---

## Entry Points

- `app_streamlit.py`  
  Streamlit user interface

- `infer_single.py`  
  CLI inference tool (developer usage)

- `train.py`  
  Model training pipeline

- `evaluate.py`  
  Model evaluation on test dataset

---

## Configuration

All configurable parameters are centralized in `config.py`:
- Active classes
- Image size
- Batch size
- Learning rates
- Backbone selection
- Paths for models and reports

---

## Reproducibility

- Dependencies locked via `requirements.txt`
- Saved model artifact: `models/best_model.keras`
- Evaluation artifacts stored in `reports/`
- Training logs saved as JSON

---

## Data Structure

```
data_split/
├─ train/
├─ val/
└─ test/
```

Each class is represented by its own subfolder.

---

## Maintenance Notes

- Keep class order consistent across training and inference.
- If the backbone is changed, ensure preprocessing compatibility.
- Consider adding face alignment for improved robustness.
- The modular structure supports future extension and refactoring.

---

## Company Scenario

The project was designed as a deployable solution requested by a marketing department.  
The documentation and repository structure allow future team members to understand, run, and extend the system within a configuration management environment.
