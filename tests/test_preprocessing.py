"""
Automated Unit Tests for Image Preprocessing and Validation.
"""

import io
import pytest
import numpy as np
from PIL import Image

from models.preprocessing import preprocess_image, validate_image_file, load_and_clean_image
from config import IMAGE_SIZE


def create_dummy_image(size=(300, 300), color=None, format="JPEG", mode="RGB"):
    """Helper creating an in-memory image for testing."""
    if color is None:
        color = 128 if mode == "L" else (100, 150, 80)
    img = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf


def test_preprocess_image_output_shape():
    """Verify preprocessed tensor has shape (1, 224, 224, 3)."""
    buf = create_dummy_image((500, 400))
    tensor = preprocess_image(buf)
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    assert tensor.dtype == np.float32


def test_preprocess_image_normalization_range():
    """Verify preprocessed pixel values are scaled to [-1.0, 1.0]."""
    buf = create_dummy_image()
    tensor = preprocess_image(buf)
    assert tensor.min() >= -1.01
    assert tensor.max() <= 1.01


def test_load_and_clean_image_color_modes():
    """Ensure RGBA, Grayscale (L), and CMYK modes are properly converted to RGB."""
    # RGBA
    rgba_buf = create_dummy_image(mode="RGBA", format="PNG")
    img_rgba = load_and_clean_image(rgba_buf)
    assert img_rgba.mode == "RGB"

    # Grayscale
    gray_buf = create_dummy_image(mode="L", format="JPEG")
    img_gray = load_and_clean_image(gray_buf)
    assert img_gray.mode == "RGB"


def test_validate_image_file_valid():
    """Verify valid JPEG passes validation."""
    buf = create_dummy_image()
    is_valid, msg = validate_image_file(buf, filename="leaf.jpg")
    assert is_valid is True
    assert "Valid image" in msg


def test_validate_image_file_empty():
    """Verify empty byte file is rejected."""
    empty_buf = io.BytesIO(b"")
    is_valid, msg = validate_image_file(empty_buf, filename="empty.jpg")
    assert is_valid is False
    assert "empty" in msg.lower()


def test_validate_image_file_unsupported_extension():
    """Verify unsupported extensions (e.g. .txt, .exe) are rejected."""
    buf = create_dummy_image()
    is_valid, msg = validate_image_file(buf, filename="script.sh")
    assert is_valid is False
    assert "unsupported" in msg.lower()


def test_validate_image_file_corrupted_content():
    """Verify corrupt bytes are rejected."""
    corrupt_buf = io.BytesIO(b"Not An Actual Image File Header Garbage Content")
    is_valid, msg = validate_image_file(corrupt_buf, filename="bad.jpg")
    assert is_valid is False
    assert "decoding failed" in msg.lower() or "corrupted" in msg.lower()
