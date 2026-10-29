import os
from pathlib import Path

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent

# Application Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-crop-disease-secret-key-2026")
DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
PORT = int(os.environ.get("PORT", 5000))
HOST = os.environ.get("HOST", "127.0.0.1")

# Upload Configuration
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB maximum upload size

# Artifacts & Model Paths
# artifacts_full/ contains the MobileNetV2 model trained on the full 54k PlantVillage dataset
ARTIFACTS_DIR = BASE_DIR / "artifacts_full"
ARTIFACTS_BASELINE_DIR = BASE_DIR / "artifacts"  # legacy baseline (untouched)
MODEL_PATH = ARTIFACTS_DIR / "crop_disease_model.keras"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
TRAINING_HISTORY_PATH = ARTIFACTS_DIR / "training_history.json"
EVALUATION_METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"

# Dataset Paths
DATA_DIR = BASE_DIR / "data"
DATASET_RAW_DIR = Path(os.environ.get("DATASET_PATH", str(DATA_DIR / "PlantVillage")))
DATASET_SPLIT_DIR = DATA_DIR / "processed"

# Model Hyperparameters & Defaults
IMAGE_SIZE = (224, 224)  # MobileNetV2 default resolution
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
DEFAULT_EPOCHS = int(os.environ.get("EPOCHS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-4))
RANDOM_SEED = 42

# Knowledge Base Path
KNOWLEDGE_BASE_PATH = BASE_DIR / "disease_info" / "knowledge_base.json"

# Ensure runtime directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
