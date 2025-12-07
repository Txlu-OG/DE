# Sentiment140 Logistic Regression API

## Dataset

- Dataset: Sentiment140 (`training.1600000.processed.noemoticon.csv`) with 1.6M tweets. [web:19][web:68]
- Labels: `polarity` column where 0 = negative, 4 = positive (neutral 2 exists in the original but this project uses binary classification). [web:20][web:26]

## Preprocessing and Model

- Preprocessing:
  - Remove URLs, @mentions, and non-alphanumeric characters.
  - Convert to lowercase and compress whitespace.
- Model:
  - TF-IDF vectorizer + Logistic Regression classifier implemented with scikit-learn. [web:69][web:72]
  - Training and evaluation done in `Project2.ipynb` using 70/30 train/validation split.
  - Reported metrics: accuracy and weighted F1-score on validation set.

## Flask API

- Implemented in `project2.py`.
- Endpoints:
  - `GET /health` → `{"status": "ok"}` if the service is running.
  - `POST /predict` with JSON body `{"text": "I love this course!"}`:
    - Cleans the text with the same preprocessing as training.
    - Uses the saved pipeline `sentiment_logreg_pipeline2.pkl` to predict.
    - Returns JSON including `prediction` (0 or 4) and `label` ("negative" or "positive").

## Docker

- Dockerfile builds an image containing:
  - Python 3.11-slim.
  - `project2.py` and `sentiment_logreg_pipeline2.pkl`.
  - Dependencies: Flask, joblib, scikit-learn. [web:28][web:73]
- Build and run locally:

