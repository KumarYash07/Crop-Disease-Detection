"""
Disease Information Package
Provides the knowledge base and lookup helpers for crop diseases.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
DEFAULT_KB_PATH = Path(__file__).resolve().parent / "knowledge_base.json"

_CACHED_KB: Optional[Dict[str, Any]] = None


def load_knowledge_base(kb_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads and caches the disease knowledge base from JSON.
    """
    global _CACHED_KB
    target_path = Path(kb_path) if kb_path else DEFAULT_KB_PATH

    if _CACHED_KB is not None and not kb_path:
        return _CACHED_KB

    try:
        if not target_path.exists():
            logger.warning(f"Knowledge base file not found at {target_path}. Using fallback.")
            return {}

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not kb_path:
                _CACHED_KB = data
            return data
    except Exception as e:
        logger.error(f"Failed to load knowledge base from {target_path}: {e}")
        return {}


def get_disease_info(class_name: str, kb: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Retrieves disease information for a given class name.
    If the class is not found in the knowledge base, returns a safe fallback dictionary.
    """
    if kb is None:
        kb = load_knowledge_base()

    if class_name in kb:
        info = dict(kb[class_name])
        info["class_key"] = class_name
        return info

    # Try heuristic fallback if format matches Crop___Disease
    if "___" in class_name:
        parts = class_name.split("___")
        crop_raw = parts[0].replace("_", " ").title()
        disease_raw = parts[1].replace("_", " ").title()
        is_healthy = "healthy" in disease_raw.lower()
    else:
        crop_raw = "Unknown Crop"
        disease_raw = class_name.replace("_", " ").title()
        is_healthy = "healthy" in disease_raw.lower()

    return {
        "class_key": class_name,
        "crop": crop_raw,
        "disease": "Healthy Plant Tissue" if is_healthy else disease_raw,
        "is_healthy": is_healthy,
        "description": f"No detailed agronomic profile is currently indexed for '{class_name}'.",
        "symptoms": [
            "Specific symptomatic visual profile not cataloged in the local database.",
            "Visual examination by an agronomist or extension officer is advised."
        ] if not is_healthy else ["Uniform green pigmentation, absence of chlorosis, necrosis, or foliar lesions."],
        "prevention": [
            "Follow standard agricultural sanitation and soil testing practices.",
            "Use certified disease-free seeds and balanced N-P-K fertilization."
        ],
        "management": [
            "Monitor plant health periodically.",
            "Consult a localized agricultural extension expert for precise diagnostic confirmation."
        ],
        "disclaimer": "Automated prediction provided for educational screening. Consult agricultural specialists for commercial treatment."
    }
