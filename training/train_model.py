"""
Model Training Script for Crop Disease Detection.
Implements MobileNetV2 Transfer Learning (with Scratch CNN fallback),
data augmentation, two-stage fine-tuning, callbacks (EarlyStopping, ModelCheckpoint),
and exports saved artifacts (model, class_names, training_history).
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# Add root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    ARTIFACTS_DIR,
    MODEL_PATH,
    CLASS_NAMES_PATH,
    TRAINING_HISTORY_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    DEFAULT_EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_mobilenetv2_model(num_classes: int, input_shape=(224, 224, 3), learning_rate=1e-4):
    """
    Constructs a transfer learning model based on MobileNetV2.
    If ImageNet weights fail to download, falls back to a custom scratch CNN.
    """
    try:
        logger.info("Attempting to load MobileNetV2 with ImageNet pretrained weights...")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False  # Freeze initial feature extractor
        architecture_type = "MobileNetV2 (Transfer Learning - Pretrained ImageNet)"
    except Exception as e:
        logger.warning(
            f"Could not download/load ImageNet weights ({e}). "
            "Falling back to custom CNN architecture trained from scratch."
        )
        return build_custom_cnn(num_classes, input_shape, learning_rate), "Custom CNN (From Scratch)"

    inputs = layers.Input(shape=input_shape, name="input_image")
    # Base feature extractor
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.4, name="dropout_0.4")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = models.Model(inputs, outputs, name="CropDisease_MobileNetV2")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, architecture_type


def build_custom_cnn(num_classes: int, input_shape=(224, 224, 3), learning_rate=1e-4):
    """
    Custom 4-block CNN architecture fallback for offline or custom training.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ], name="CropDisease_CustomCNN")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def create_dataset_pipeline(image_paths, labels, batch_size=BATCH_SIZE, is_training=True):
    """
    TensorFlow tf.data pipeline with MobileNetV2 preprocessing and data augmentation.
    """
    def load_and_preprocess(path_tensor, label_tensor):
        img_raw = tf.io.read_file(path_tensor)
        img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, IMAGE_SIZE)
        # MobileNetV2 standard normalization: [-1, 1]
        img = (tf.cast(img, tf.float32) / 127.5) - 1.0
        return img, label_tensor

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=min(len(image_paths), 2000), seed=RANDOM_SEED)

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    parser = argparse.ArgumentParser(description="Train CNN model for crop disease classification.")
    parser.add_argument("--manifest", type=str, default=str(BASE_DIR / "data" / "processed" / "splits_manifest.json"),
                        help="Path to splits manifest json.")
    parser.add_argument("--artifacts-dir", type=str, default=str(ARTIFACTS_DIR), help="Directory to save model weights and history.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Initial learning rate.")
    parser.add_argument("--fine-tune", action="store_true", help="Enable stage-2 fine-tuning of top layers.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    artifacts_target = Path(args.artifacts_dir)
    artifacts_target.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        logger.error(
            f"Splits manifest not found at '{manifest_path}'.\n"
            "Please ensure the dataset is downloaded and prepared."
        )
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Class indexing
    class_names = sorted(list(splits["train"].keys()))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    num_classes = len(class_names)
    logger.info(f"Loaded {num_classes} classes from manifest.")

    # Flatten file paths and labels (resolving relative paths against BASE_DIR)
    train_paths, train_labels = [], []
    for c, files in splits["train"].items():
        resolved_files = [str((BASE_DIR / p).resolve()) if not Path(p).is_absolute() else str(p) for p in files]
        train_paths.extend(resolved_files)
        train_labels.extend([class_to_idx[c]] * len(files))

    val_paths, val_labels = [], []
    for c, files in splits["val"].items():
        resolved_files = [str((BASE_DIR / p).resolve()) if not Path(p).is_absolute() else str(p) for p in files]
        val_paths.extend(resolved_files)
        val_labels.extend([class_to_idx[c]] * len(files))

    logger.info(f"Dataset split size: {len(train_paths)} Train samples, {len(val_paths)} Validation samples.")

    # Build datasets
    train_ds = create_dataset_pipeline(train_paths, train_labels, batch_size=args.batch_size, is_training=True)
    val_ds = create_dataset_pipeline(val_paths, val_labels, batch_size=args.batch_size, is_training=False)

    # Build model
    model, arch_type = build_mobilenetv2_model(num_classes, input_shape=(*IMAGE_SIZE, 3), learning_rate=args.lr)
    logger.info(f"Initialized model architecture: {arch_type}")
    model.summary(print_fn=logger.info)

    model_target_path = artifacts_target / "crop_disease_model.keras"
    class_names_target = artifacts_target / "class_names.json"
    history_target = artifacts_target / "training_history.json"

    # Setup callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(model_target_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    early_stop_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    # Stage 1: Feature Extraction
    logger.info(f"Starting training phase 1 for {args.epochs} epochs with batch size {args.batch_size}...")
    start_time = time.time()
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb]
    )

    combined_history = {k: [float(val) for val in v] for k, v in history_phase1.history.items()}

    # Optional Stage 2: Fine-Tuning
    if args.fine_tune and "MobileNetV2" in arch_type:
        logger.info("Starting training phase 2: Unfreezing top 30 layers for fine-tuning...")
        base_layer = model.get_layer("mobilenetv2_1.00_224") if "mobilenetv2_1.00_224" in [l.name for l in model.layers] else model.layers[1]
        base_layer.trainable = True
        for layer in base_layer.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=optimizers.Adam(learning_rate=args.lr / 10.0),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max(3, args.epochs // 2),
            callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb]
        )

        for k, v in history_phase2.history.items():
            combined_history[k].extend([float(val) for val in v])

    total_training_time = time.time() - start_time
    combined_history["total_training_time_seconds"] = round(total_training_time, 2)
    combined_history["total_training_time_formatted"] = f"{int(total_training_time // 60)}m {int(total_training_time % 60)}s"

    logger.info(f"Training completed in {combined_history['total_training_time_formatted']} ({total_training_time:.2f}s).")

    # Save artifacts
    with open(class_names_target, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    logger.info(f"Saved class labels to: {class_names_target}")

    with open(history_target, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=2)
    logger.info(f"Saved training history to: {history_target}")

    # Ensure best model is saved
    if not model_target_path.exists():
        model.save(str(model_target_path))
    logger.info(f"Best model successfully saved at: {model_target_path}")
    logger.info("Training stage completed.")


if __name__ == "__main__":
    main()
