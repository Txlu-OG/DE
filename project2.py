"""
project2.py

Flask API for Sentiment140 logistic regression model.

This script is the *deployment* side of the project:
- It DOES NOT train the model (training was done in Jupyter Notebook on the full Sentiment140 file).
- It loads the already trained TF-IDF + Logistic Regression pipeline
  saved as 'sentiment_logreg_pipeline2.pkl'.
- It exposes a Flask API with endpoints:
    /health   -> to check service status
    /predict  -> to send a tweet and get sentiment back
- It is intended to be run as:
    python project.py --mode api
  or inside Docker with:
    CMD ["python", "project.py", "--mode", "api"]
"""

import argparse
import os
import re

import joblib
from flask import Flask, request, jsonify

# ======================================================
# 1. Configuration
# ======================================================

# Base directory where this script and the model file live
BASE_DIR = os.path.dirname(__file__)

# DIFFERENCE vs old 150k model:
#   - OLD: pointed to 'sentiment_logreg_pipeline.pkl' (subset-trained)
#   - NEW: points to 'sentiment_logreg_pipeline2.pkl' (full-data notebook model)
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_logreg_pipeline2.pkl")

# Initialize Flask app
app = Flask(__name__)

# Global variable for the loaded model pipeline
model_pipeline = None

# For this project the notebook trained a binary model:
# 0 = negative, 4 = positive (Sentiment140 labels). [web:19][web:20]
LABEL_MAP = {
    0: "negative",
    4: "positive"
}

# ======================================================
# 2. Preprocessing (must match notebook)
# ======================================================

def clean_tweet(text: str) -> str:
    """
    Same cleaning logic used in the Jupyter notebook:
    - Remove URLs
    - Remove @mentions
    - Remove non-alphanumeric characters
    - Lowercase and compress spaces

    Using identical preprocessing at inference time ensures that the
    model sees data in the same format as during training. [web:22][web:72]
    """
    if not isinstance(text, str):
        text = str(text)

    # Remove URLs (http, https, www)
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove non-alphanumeric characters (keep letters, digits, spaces)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)

    # Lowercase
    text = text.lower()

    # Compress multiple spaces into one and strip edges
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ======================================================
# 3. Model loading
# ======================================================

def load_model():
    """
    Load the trained pipeline (TF-IDF + Logistic Regression) from disk.

    The notebook created and saved this pipeline using joblib on the
    *full* Sentiment140 dataset, not just the 150k subset. [web:19][web:68]

    We cache it in a global variable so we don't reload it on every request.
    """
    global model_pipeline

    if model_pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                f"Make sure you ran the Jupyter notebook to train and save the model first."
            )
        model_pipeline = joblib.load(MODEL_PATH)

    return model_pipeline

# ======================================================
# 4. Flask endpoints
# ======================================================

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    Returns 200 OK if the service is up.

    Useful for:
    - Manual checks
    - Docker health checks
    - Kubernetes liveness/readiness probes
    """
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint.

    Expects JSON with a 'text' field, for example:
        {
            "text": "I love this course!"
        }

    Steps:
    1. Validate input JSON.
    2. Clean the text with the same function used during training.
    3. Use the loaded pipeline to predict the sentiment label (0 or 4).
    4. Map the numeric label to a human-readable string (negative/positive).
    5. Return the result as JSON.
    """
    # Parse JSON body
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    # Validate required field
    if not payload or "text" not in payload:
        return jsonify({"error": "JSON body must contain 'text' field"}), 400

    raw_text = str(payload["text"])

    # Clean the input text the same way as the training data
    cleaned_text = clean_tweet(raw_text)

    # Load (or get cached) model pipeline
    model = load_model()

    # Pipeline expects a list/array of documents
    pred_array = model.predict([cleaned_text])
    pred_int = int(pred_array[0])

    # Map numeric label to human-readable sentiment
    label = LABEL_MAP.get(pred_int, "unknown")

    response = {
        "input_text": raw_text,
        "cleaned_text": cleaned_text,
        "prediction": pred_int,
        "label": label
    }

    return jsonify(response), 200

# ======================================================
# 5. Runner
# ======================================================

def run_api():
    """
    Start the Flask development server on 0.0.0.0:5000.

    Used when:
    - You run `python project.py --mode api` in PyCharm or PowerShell.
    - Docker container starts with CMD ["python", "project.py", "--mode", "api"].
    """
    # Fail fast if the model file is missing
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file {MODEL_PATH} not found. "
            f"Train and save the model in Jupyter notebook before starting the API."
        )

    print("Loading model and starting Flask API on http://0.0.0.0:5000 ...")
    load_model()
    app.run(host="0.0.0.0", port=5000)

# ======================================================
# 6. CLI argument parsing
# ======================================================

def parse_args():
    """
    Command-line interface for the script.

    For this deployment script we only support:
      --mode api : run the Flask prediction API

    Keeping the interface explicit makes it easy to plug into Docker and CI/CD:
    - Local:  python project.py --mode api
    - Docker: CMD ["python", "project.py", "--mode", "api"] [web:28][web:79]
    """
    parser = argparse.ArgumentParser(description="Sentiment140 Flask API")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["api"],
        required=True,
        help="api = run the Flask prediction API"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "api":
        run_api()
