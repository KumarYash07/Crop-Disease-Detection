"""
Automated Integration Tests for Flask API Endpoints.
"""

import io
import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def create_dummy_image():
    img = Image.new("RGB", (224, 224), color=(60, 140, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_index_page(client):
    """GET / should serve home dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Crop Disease Detection" in response.data


def test_diseases_page(client):
    """GET /diseases should render supported diseases catalog."""
    response = client.get("/diseases")
    assert response.status_code == 200
    assert b"Supported Crops" in response.data


def test_about_page(client):
    """GET /about should render architecture documentation."""
    response = client.get("/about")
    assert response.status_code == 200
    assert b"MobileNetV2" in response.data


def test_api_health_endpoint(client):
    """GET /api/health returns application and model diagnostics."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    assert "model_loaded" in data
    assert "num_classes" in data
    assert data["status"] == "ok"


def test_api_classes_endpoint(client):
    """GET /api/classes returns class list."""
    response = client.get("/api/classes")
    assert response.status_code == 200
    data = response.get_json()
    assert "total_classes" in data
    assert "classes" in data


def test_api_predict_missing_file(client):
    """POST /api/predict without 'image' multipart field returns 400."""
    with patch("app.is_model_available", return_value=True):
        response = client.post("/api/predict", data={})
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "No image file provided" in data["error"]


def test_api_predict_empty_filename(client):
    """POST /api/predict with empty filename returns 400."""
    with patch("app.is_model_available", return_value=True):
        data = {"image": (io.BytesIO(b""), "")}
        response = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert response.status_code == 400
        res_json = response.get_json()
        assert res_json["success"] is False
        assert "Empty filename" in res_json["error"]


def test_api_predict_corrupted_image(client):
    """POST /api/predict with invalid image content returns 400."""
    with patch("app.is_model_available", return_value=True):
        corrupt_data = {"image": (io.BytesIO(b"Not an actual image"), "leaf.jpg")}
        response = client.post("/api/predict", data=corrupt_data, content_type="multipart/form-data")
        assert response.status_code == 400
        res_json = response.get_json()
        assert res_json["success"] is False
        assert "Invalid image file" in res_json["error"]


def test_api_predict_when_model_unavailable(client):
    """POST /api/predict returns 503 if model is not loaded."""
    with patch("app.is_model_available", return_value=False):
        data = {"image": (create_dummy_image(), "leaf.jpg")}
        response = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert response.status_code == 503
        res_json = response.get_json()
        assert res_json["success"] is False
        assert "unavailable" in res_json["error"].lower()


def test_api_predict_successful_mock_flow(client):
    """POST /api/predict returns 200 and schema when model succeeds."""
    mock_result = {
        "success": True,
        "predicted_class": "Tomato___Early_blight",
        "display_name": "Tomato - Early Blight",
        "crop": "Tomato",
        "disease": "Early Blight",
        "is_healthy": False,
        "confidence": 0.945,
        "confidence_percentage": 94.5,
        "top_predictions": [
            {"rank": 1, "class_name": "Tomato___Early_blight", "confidence_percent": 94.5}
        ],
        "disease_information": {
            "crop": "Tomato",
            "disease": "Early Blight",
            "symptoms": ["Dark concentric spots"],
            "prevention": ["Crop rotation"],
            "management": ["Fungicide"]
        },
        "disclaimer": "Educational disclaimer"
    }

    with patch("app.is_model_available", return_value=True):
        with patch("app.predict_crop_disease", return_value=mock_result):
            data = {"image": (create_dummy_image(), "tomato_leaf.jpg")}
            response = client.post("/api/predict", data=data, content_type="multipart/form-data")
            assert response.status_code == 200
            res_json = response.get_json()
            assert res_json["success"] is True
            assert res_json["predicted_class"] == "Tomato___Early_blight"
            assert res_json["confidence_percentage"] == 94.5
