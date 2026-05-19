from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    assert client.get("/health").status_code == 200

def test_predict_penguin():
    payload = {
        "species": "Adelie",
        "island": "Torgersen",
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "sex": "Male"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_body_mass_g" in response.json()