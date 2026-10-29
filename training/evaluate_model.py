"""
Model Evaluation Script for Crop Disease Detection.
Evaluates the trained CNN on the held-out test split, computes classification metrics
(Accuracy, Macro/Weighted Precision, Recall, F1-score), saves a detailed JSON report,
and plots the normalized confusion matrix and training curves.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import tensorflow as tf

# Add root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    ARTIFACTS_DIR,
    MODEL_PATH,
    CLASS_NAMES_PATH,
    TRAINING_HISTORY_PATH,
    EVALUATION_METRICS_PATH,
    IMAGE_SIZE,
    BATCH_SIZE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def plot_training_curves(history_path: Path, output_image_path: Path):
    """
    Renders loss and accuracy progression across training and validation epochs.
    """
    if not history_path.exists():
        logger.warning(f"Training history file '{history_path}' not found. Skipping curve plotting.")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history.get("loss", [])) + 1)
    if len(epochs) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history.get("loss", []), "o-", label="Training Loss", color="#d32f2f")
    if "val_loss" in history:
        ax1.plot(epochs, history.get("val_loss", []), "s--", label="Validation Loss", color="#f57c00")
    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss (Sparse Categorical Crossentropy)", fontsize=10)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy curve
    ax2.plot(epochs, history.get("accuracy", []), "o-", label="Training Accuracy", color="#2e7d32")
    if "val_accuracy" in history:
        ax2.plot(epochs, history.get("val_accuracy", []), "s--", label="Validation Accuracy", color="#388e3c")
    ax2.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_image_path), dpi=200)
    plt.close()
    logger.info(f"Saved training curves plot to: {output_image_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, output_image_path: Path):
    """
    Renders a normalized confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    # Row-normalized
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(max(10, len(class_names) * 0.4), max(8, len(class_names) * 0.35)))
    clean_labels = [c.replace("___", "\n").replace("_", " ") for c in class_names]

    sns.heatmap(
        cm_norm,
        annot=len(class_names) <= 15,
        fmt=".2f",
        cmap="Greens",
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        cbar_kws={"label": "Normalized Recall Proportion"}
    )
    plt.title("Normalized Confusion Matrix on Held-Out Test Set", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_image_path), dpi=200)
    plt.close()
    logger.info(f"Saved confusion matrix plot to: {output_image_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained CNN model on held-out test split.")
    parser.add_argument("--manifest", type=str, default=str(BASE_DIR / "data" / "processed" / "splits_manifest.json"),
                        help="Path to splits manifest.")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Directory containing model artifacts to evaluate.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model .keras file.")
    parser.add_argument("--classes-path", type=str, default=None, help="Path to class names json.")
    args = parser.parse_args()

    art_dir = Path(args.artifacts_dir) if args.artifacts_dir else ARTIFACTS_DIR
    model_file = Path(args.model_path) if args.model_path else (art_dir / "crop_disease_model.keras")
    classes_file = Path(args.classes_path) if args.classes_path else (art_dir / "class_names.json")
    history_file = art_dir / "training_history.json"
    metrics_file = art_dir / "evaluation_metrics.json"

    if not model_file.exists():
        logger.error(f"Trained model not found at '{model_file}'. Train the model first.")
        sys.exit(1)

    if not classes_file.exists():
        logger.error(f"Class names mapping not found at '{classes_file}'.")
        sys.exit(1)

    with open(classes_file, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    manifest_file = Path(args.manifest)
    if not manifest_file.exists():
        logger.error(f"Dataset split manifest not found at '{manifest_file}'.")
        sys.exit(1)

    with open(manifest_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_split = splits.get("test", {})
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    test_paths, test_labels = [], []
    for c, files in test_split.items():
        if c in class_to_idx:
            resolved_files = [str((BASE_DIR / p).resolve()) if not Path(p).is_absolute() else str(p) for p in files]
            test_paths.extend(resolved_files)
            test_labels.extend([class_to_idx[c]] * len(files))

    if not test_paths:
        logger.error("No test samples found in the manifest.")
        sys.exit(1)

    logger.info(f"Evaluating model on {len(test_paths)} held-out test samples across {len(class_names)} classes.")

    # Load model
    logger.info(f"Loading model from {model_file}...")
    model = tf.keras.models.load_model(str(model_file), compile=False)

    # Build evaluation dataset
    def load_and_preprocess(path_tensor):
        img_raw = tf.io.read_file(path_tensor)
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = (img / 127.5) - 1.0
        return img

    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Inference
    logger.info("Executing batch inference across the test dataset...")
    predictions_raw = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(predictions_raw, axis=1)
    y_true = np.array(test_labels)

    # Compute metrics
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    evaluation_results = {
        "dataset": "PlantVillage (Held-out Test Split)",
        "num_test_samples": len(test_paths),
        "num_classes": len(class_names),
        "test_accuracy": round(acc, 4),
        "macro_avg": {
            "precision": round(float(p_macro), 4),
            "recall": round(float(r_macro), 4),
            "f1_score": round(float(f1_macro), 4)
        },
        "weighted_avg": {
            "precision": round(float(p_weighted), 4),
            "recall": round(float(r_weighted), 4),
            "f1_score": round(float(f1_weighted), 4)
        },
        "per_class_metrics": {
            c: {
                "precision": round(report_dict[c]["precision"], 4),
                "recall": round(report_dict[c]["recall"], 4),
                "f1_score": round(report_dict[c]["f1-score"], 4),
                "support": int(report_dict[c]["support"])
            }
            for c in class_names if c in report_dict
        }
    }

    # Save metrics JSON into target directory
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)
    logger.info(f"Saved evaluation metrics to: {metrics_file}")

    # Generate charts into target directory
    plot_training_curves(history_file, art_dir / "training_curves.png")
    plot_confusion_matrix(y_true, y_pred, class_names, art_dir / "confusion_matrix.png")

    logger.info("Evaluation complete.")
    logger.info(f"Test Accuracy: {acc * 100:.2f}% | Macro F1: {f1_macro * 100:.2f}% | Weighted F1: {f1_weighted * 100:.2f}%")


if __name__ == "__main__":
    main()
