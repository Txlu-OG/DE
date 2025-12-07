FROM python:3.11-slim

WORKDIR /app

# Copy code and model
COPY project2.py ./project2.py
COPY sentiment_logreg_pipeline2.pkl ./sentiment_logreg_pipeline2.pkl

# Install dependencies
RUN pip install --no-cache-dir flask joblib scikit-learn

# Expose Flask port
EXPOSE 5000

# Run the API
CMD ["python", "project2.py", "--mode", "api"]
