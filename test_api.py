# test_api.py
# test_api.py
# Basic tests for the Flask API using the app object from project2.py

from project2 import app

def test_health():
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}

def test_predict():
    client = app.test_client()
    resp = client.post("/predict", json={"text": "This is great!"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "label" in data
    assert data["label"] in ["positive", "negative"]