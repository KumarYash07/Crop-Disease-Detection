"""
Image Preprocessing Module for Crop Disease Detection.
Handles image validation, color mode normalization, EXIF rotation handling,
and resizing/scaling matching MobileNetV2 CNN input expectations.
"""

import io
import logging
from typing import Tuple, Union, BinaryIO
import numpy as np
from PIL import Image, ImageOps

from config import IMAGE_SIZE, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH

logger = logging.getLogger(__name__)


def validate_image_file(file_obj: Union[BinaryIO, bytes, str], filename: str = "") -> Tuple[bool, str]:
    """
    Validates that the provided file data is non-empty, within size limits,
    and a valid decodable image.
    """
    # Check filename extension if available
    if filename:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported file extension '.{ext}'. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}."

    # Inspect bytes
    if isinstance(file_obj, bytes):
        raw_bytes = file_obj
    elif hasattr(file_obj, "read"):
        raw_bytes = file_obj.read()
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
    elif isinstance(file_obj, str):
        try:
            with open(file_obj, "rb") as f:
                raw_bytes = f.read()
        except Exception as e:
            return False, f"Could not read image file from disk: {e}"
    else:
        return False, "Invalid image input type."

    if len(raw_bytes) == 0:
        return False, "Uploaded image is empty (0 bytes)."

    if len(raw_bytes) > MAX_CONTENT_LENGTH:
        return False, f"Image size exceeds the maximum limit of {MAX_CONTENT_LENGTH // (1024 * 1024)} MB."

    # Verify PIL can open and verify image integrity
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.verify()
    except Exception as e:
        return False, f"Image decoding failed or file is corrupted: {e}"

    return True, "Valid image."


def load_and_clean_image(image_input: Union[BinaryIO, bytes, str, Image.Image]) -> Image.Image:
    """
    Loads an image, corrects EXIF orientation, and converts it to standard 3-channel RGB.
    """
    if isinstance(image_input, Image.Image):
        img = image_input.copy()
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif hasattr(image_input, "read"):
        raw_bytes = image_input.read()
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        img = Image.open(io.BytesIO(raw_bytes))
    elif isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Auto-orient based on EXIF tags (e.g. mobile camera orientation)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        logger.debug(f"EXIF transposition skipped: {e}")

    # Ensure 3-channel RGB (handle RGBA, CMYK, Grayscale, P-mode)
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def preprocess_image(
    image_input: Union[BinaryIO, bytes, str, Image.Image],
    target_size: Tuple[int, int] = IMAGE_SIZE
) -> np.ndarray:
    """
    Preprocesses an image for CNN inference.
    1. Loads and corrects EXIF orientation and RGB channels.
    2. Resizes to target_size (default 224x224).
    3. Normalizes pixel values to [-1.0, 1.0] matching MobileNetV2 standard:
       pixel_norm = (pixel / 127.5) - 1.0
    4. Expands dimensions to produce batch shape: (1, height, width, 3).
    """
    img = load_and_clean_image(image_input)

    # Resize using high-quality bilinear interpolation
    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to float32 NumPy array
    img_array = np.asarray(img, dtype=np.float32)

    # MobileNetV2 normalizes RGB values from [0, 255] to [-1, 1]
    # (Exact equivalent of tf.keras.applications.mobilenet_v2.preprocess_input)
    img_array = (img_array / 127.5) - 1.0

    # Expand batch dimension: (1, 224, 224, 3)
    batch_array = np.expand_dims(img_array, axis=0)
    return batch_array
