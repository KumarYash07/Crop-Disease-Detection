"""
Flask Backend Application for Crop Disease Detection.
Serves interactive web dashboard, health diagnostics, dynamic class listing,
and real-time CNN inference endpoints with comprehensive error handling.
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from models.model_loader import get_model, get_class_names, is_model_available, get_model_status
from models.preprocessing import validate_image_file
from models.predictor import predict_crop_disease
from disease_info import load_knowledge_base, get_disease_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CropDiseaseApp")

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER

# Warm-up / load model and knowledge base once at startup
logger.info("Initializing Crop Disease Detection Backend...")
_ = load_knowledge_base()
_ = get_model()
_ = get_class_names()
logger.info(f"Startup check: Model available={is_model_available()}, Classes count={len(get_class_names())}")


# --- HTML Page Routes ---

@app.route("/")
def index():
    """Renders the main Crop Disease Detection dashboard."""
    classes = get_class_names()
    model_status = get_model_status()
    kb = load_knowledge_base()
    return render_template(
        "index.html",
        classes=classes,
        model_status=model_status,
        knowledge_base=kb
    )


@app.route("/diseases")
def diseases():
    """Renders the directory of supported crops, diseases, and management guidance."""
    classes = get_class_names()
    kb = load_knowledge_base()

    # If classes are not yet populated from a trained model, use keys from knowledge_base
    display_classes = classes if classes else sorted(list(kb.keys()))

    # Group classes by crop for clean academic presentation
    grouped_diseases = {}
    for cname in display_classes:
        info = get_disease_info(cname, kb)
        crop = info.get("crop", "Other")
        if crop not in grouped_diseases:
            grouped_diseases[crop] = []
        grouped_diseases[crop].append({
            "class_key": cname,
            "disease": info.get("disease", cname),
            "is_healthy": info.get("is_healthy", False),
            "description": info.get("description", ""),
            "symptoms": info.get("symptoms", []),
            "prevention": info.get("prevention", []),
            "management": info.get("management", [])
        })

    return render_template(
        "diseases.html",
        grouped_diseases=grouped_diseases,
        total_crops=len(grouped_diseases),
        total_diseases=len(display_classes)
    )


@app.route("/about")
def about():
    """Renders the about page describing CNN architecture, methodology, and limitations."""
    model_status = get_model_status()
    return render_template(
        "about.html",
        model_status=model_status
    )


# --- API Routes ---

@app.route("/api/health", methods=["GET"])
def api_health():
    """
    Returns application and model-loading diagnostic status.
    Reflects the actual system state without hardcoding.
    """
    status = get_model_status()
    return jsonify({
        "status": "ok",
        "model_loaded": status["model_loaded"],
        "num_classes": status["num_classes"],
        "model_path": status["model_path"],
        "model_exists_on_disk": status["model_exists_on_disk"],
        "classes_exist_on_disk": status["classes_exist_on_disk"],
        "input_shape": status["input_shape"],
        "error": status["error"]
    }), 200


@app.route("/api/classes", methods=["GET"])
def api_classes():
    """
    Returns the list of actual disease class names loaded from the model artifacts.
    """
    classes = get_class_names()
    return jsonify({
        "total_classes": len(classes),
        "classes": classes,
        "classes_loaded": len(classes) > 0
    }), 200


@app.route("/api/disease/<class_name>", methods=["GET"])
def api_disease_detail(class_name: str):
    """
    Returns agronomic profile and management guidance for a given class name.
    """
    kb = load_knowledge_base()
    info = get_disease_info(class_name, kb)
    return jsonify(info), 200


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts multipart/form-data image upload and executes CNN inference.
    """
    # 1. Verify model availability
    if not is_model_available():
        status = get_model_status()
        return jsonify({
            "success": False,
            "error": "CNN model is currently unavailable on the server.",
            "details": status.get("error") or "Model artifact has not been trained or placed in the artifacts directory.",
            "instructions": "Run 'python training/train_model.py' to produce 'artifacts/crop_disease_model.keras'."
        }), 503

    # 2. Verify file presence
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided in the request.",
            "details": "Expected a multipart/form-data request with key 'image'."
        }), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "Empty filename submitted.",
            "details": "Please select a valid image file before analyzing."
        }), 400

    # 3. Validate image file extension and decodability
    filename = secure_filename(file.filename)
    is_valid, validation_msg = validate_image_file(file, filename=filename)
    if not is_valid:
        return jsonify({
            "success": False,
            "error": "Invalid image file.",
            "details": validation_msg
        }), 400

    # 4. Perform actual CNN inference
    try:
        file.seek(0)
        prediction_result = predict_crop_disease(file, top_k=3)
        return jsonify(prediction_result), 200
    except ValueError as ve:
        logger.error(f"Image processing error during inference: {ve}")
        return jsonify({
            "success": False,
            "error": "Image preprocessing failed.",
            "details": str(ve)
        }), 400
    except Exception as e:
        logger.error(f"Inference execution failure: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "An internal error occurred during model inference.",
            "details": "Please verify that the input image is an uncorrupted plant photograph."
        }), 500


# --- Global Error Handlers ---

@app.errorhandler(413)
def handle_large_file(e):
    return jsonify({
        "success": False,
        "error": "File size exceeds the allowed limit.",
        "details": f"Maximum permitted image upload is {config.MAX_CONTENT_LENGTH // (1024 * 1024)} MB."
    }), 413


@app.errorhandler(404)
def handle_not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "API endpoint not found."}), 404
    return render_template("base.html", title="Page Not Found"), 404


@app.errorhandler(500)
def handle_server_error(e):
    logger.error(f"Internal server error: {e}")
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Internal server error occurred."}), 500
    return render_template("base.html", title="Server Error"), 500


if __name__ == "__main__":
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )
