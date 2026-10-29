"""
Automated Unit Tests for Model Predictor and Disease Knowledge Base.
"""

import io
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

from models.predictor import predict_crop_disease
from disease_info import get_disease_info, load_knowledge_base


def create_dummy_image():
    img = Image.new("RGB", (224, 224), color=(50, 120, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_missing_model_raises_runtime_error():
    """Predictor must raise RuntimeError when model is not available."""
    buf = create_dummy_image()
    with patch("models.predictor.get_model", return_value=None):
        with pytest.raises(RuntimeError, match="not loaded"):
            predict_crop_disease(buf, class_names=["Apple___Apple_scab"])


def test_missing_class_names_raises_runtime_error():
    """Predictor must raise RuntimeError when class list is empty."""
    buf = create_dummy_image()
    mock_model = MagicMock()
    with pytest.raises(RuntimeError, match="mapping is missing"):
        predict_crop_disease(buf, model=mock_model, class_names=[])


def test_predictor_output_structure_with_mock_model():
    """Verify inference output matches the expected JSON response schema."""
    buf = create_dummy_image()
    classes = ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"]

    # Mock model outputting specific softmax probabilities
    mock_model = MagicMock()
    # Let's say Tomato___Early_blight has 0.85, Late_blight 0.10, healthy 0.05
    mock_model.predict.return_value = np.array([[0.85, 0.10, 0.05]])

    result = predict_crop_disease(buf, top_k=3, model=mock_model, class_names=classes)

    assert result["success"] is True
    assert result["predicted_class"] == "Tomato___Early_blight"
    assert result["crop"] == "Tomato"
    assert result["disease"] == "Early Blight"
    assert result["is_healthy"] is False
    assert result["confidence"] == 0.85
    assert result["confidence_percentage"] == 85.0
    assert len(result["top_predictions"]) == 3
    assert result["top_predictions"][0]["class_name"] == "Tomato___Early_blight"
    assert result["top_predictions"][0]["rank"] == 1
    assert "disease_information" in result
    assert "disclaimer" in result


def test_knowledge_base_lookup_existing():
    """Ensure cataloged disease returns rich agronomic profile."""
    kb = load_knowledge_base()
    info = get_disease_info("Potato___Early_blight", kb)
    assert info["crop"] == "Potato"
    assert info["disease"] == "Early Blight"
    assert info["is_healthy"] is False
    assert len(info["symptoms"]) > 0
    assert len(info["prevention"]) > 0
    assert len(info["management"]) > 0


def test_knowledge_base_lookup_unknown_fallback():
    """Ensure uncataloged or custom class generates a safe fallback profile."""
    kb = load_knowledge_base()
    info = get_disease_info("CustomCrop___Unknown_virus", kb)
    assert info["crop"] == "Customcrop"
    assert "virus" in info["disease"].lower()
    assert len(info["symptoms"]) > 0
    assert "extension" in info["management"][1].lower()
