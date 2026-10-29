"""
Model Loader Module for Crop Disease Detection.
Provides safe caching and initialization of the trained Keras CNN model and class mapping.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Any, Dict
import tensorflow as tf

import config

logger = logging.getLogger(__name__)

_CACHED_MODEL: Optional[tf.keras.Model] = None
_CACHED_CLASSES: Optional[List[str]] = None
_MODEL_LOAD_ERROR: Optional[str] = None


def get_model(model_path: Optional[Path] = None, force_reload: bool = False) -> Optional[tf.keras.Model]:
    """
    Returns the cached Keras CNN model. Loads from disk if not already loaded.
    If the model file does not exist, returns None and logs a clear warning.
    """
    global _CACHED_MODEL, _MODEL_LOAD_ERROR
    target_path = Path(model_path) if model_path else config.MODEL_PATH

    if _CACHED_MODEL is not None and not force_reload and not model_path:
        return _CACHED_MODEL

    if not target_path.exists():
        msg = f"Trained model artifact not found at '{target_path}'. Please run training/train_model.py first."
        _MODEL_LOAD_ERROR = msg
        logger.warning(msg)
        return None

    try:
        logger.info(f"Loading trained CNN model from {target_path}...")
        model = tf.keras.models.load_model(str(target_path), compile=False)
        if not model_path:
            _CACHED_MODEL = model
            _MODEL_LOAD_ERROR = None
        logger.info("CNN model loaded successfully into memory.")
        return model
    except Exception as e:
        err_msg = f"Failed to load CNN model from '{target_path}': {str(e)}"
        _MODEL_LOAD_ERROR = err_msg
        logger.error(err_msg, exc_info=True)
        return None


def get_class_names(class_path: Optional[Path] = None, force_reload: bool = False) -> List[str]:
    """
    Loads and caches the list of class labels saved alongside the model.
    """
    global _CACHED_CLASSES
    target_path = Path(class_path) if class_path else config.CLASS_NAMES_PATH

    if _CACHED_CLASSES is not None and not force_reload and not class_path:
        return _CACHED_CLASSES

    if not target_path.exists():
        logger.warning(f"Class names mapping file not found at '{target_path}'.")
        return []

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            classes = json.load(f)
            if isinstance(classes, list):
                if not class_path:
                    _CACHED_CLASSES = classes
                return classes
            else:
                logger.error(f"Invalid format in {target_path}, expected JSON list.")
                return []
    except Exception as e:
        logger.error(f"Error loading class names from {target_path}: {e}")
        return []


def is_model_available() -> bool:
    """
    Quick status check determining whether both model and class mappings are loaded.
    """
    model = get_model()
    classes = get_class_names()
    return (model is not None) and (len(classes) > 0)


def get_model_status() -> Dict[str, Any]:
    """
    Returns an academic and operational diagnostics dictionary of the model state.
    """
    model = get_model()
    classes = get_class_names()
    return {
        "model_loaded": model is not None,
        "model_path": str(config.MODEL_PATH),
        "model_exists_on_disk": config.MODEL_PATH.exists(),
        "num_classes": len(classes),
        "classes_path": str(config.CLASS_NAMES_PATH),
        "classes_exist_on_disk": config.CLASS_NAMES_PATH.exists(),
        "input_shape": list(model.input_shape) if model is not None and hasattr(model, "input_shape") else None,
        "error": _MODEL_LOAD_ERROR
    }
