"""
Dataset Preparation Script for Crop Disease Detection (PlantVillage).
Discovers classes, filters corrupted or duplicate images, creates reproducible
70% Train / 15% Validation / 15% Test splits, and exports class mappings.
"""

import os
import sys
import json
import hashlib
import argparse
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    DATASET_RAW_DIR,
    DATASET_SPLIT_DIR,
    CLASS_NAMES_PATH,
    ARTIFACTS_DIR,
    RANDOM_SEED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def compute_file_hash(filepath: Path) -> str:
    """Computes MD5 hash for image deduplication."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_and_collect_images(raw_dir: Path):
    """
    Scans class folders, detects duplicates, and validates image readability.
    Automatically resolves nested 'raw/color' if pointing to root PlantVillage_Images.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory '{raw_dir}' does not exist!\n"
            f"Please ensure the dataset is downloaded into: {raw_dir}"
        )

    # Auto-resolve nested raw/color if present
    if (raw_dir / "raw" / "color").is_dir():
        logger.info(f"Detected nested 'raw/color' folder. Redirecting data source to '{raw_dir / 'raw' / 'color'}'.")
        raw_dir = raw_dir / "raw" / "color"
    elif (raw_dir / "color").is_dir():
        logger.info(f"Detected nested 'color' folder. Redirecting data source to '{raw_dir / 'color'}'.")
        raw_dir = raw_dir / "color"

    class_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not class_dirs:
        raise ValueError(f"No class subdirectories found inside '{raw_dir}'.")

    class_dirs.sort(key=lambda d: d.name)
    logger.info(f"Discovered {len(class_dirs)} class directories in '{raw_dir}'.")

    class_images = defaultdict(list)
    seen_hashes = {}
    total_scanned = 0
    duplicate_count = 0
    corrupt_count = 0

    for cdir in class_dirs:
        cname = cdir.name
        for img_path in cdir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                total_scanned += 1
                # Check for readable image
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception:
                    corrupt_count += 1
                    logger.debug(f"Skipping corrupted image: {img_path}")
                    continue

                # Deduplication check via MD5
                fhash = compute_file_hash(img_path)
                if fhash in seen_hashes:
                    duplicate_count += 1
                    continue
                seen_hashes[fhash] = str(img_path)

                # Store normalized path relative to BASE_DIR if inside workspace, else absolute
                try:
                    rel_path = str(img_path.resolve().relative_to(BASE_DIR)).replace("\\", "/")
                except ValueError:
                    rel_path = str(img_path.resolve()).replace("\\", "/")

                class_images[cname].append(rel_path)

    logger.info(f"Scan complete. Total files scanned: {total_scanned}")
    logger.info(f"Unique valid images: {sum(len(v) for v in class_images.values())}")
    logger.info(f"Duplicates pruned: {duplicate_count}, Corrupt images rejected: {corrupt_count}")

    # Remove any empty classes
    class_images = {k: v for k, v in class_images.items() if len(v) > 0}
    return class_images, duplicate_count, corrupt_count, total_scanned


def split_dataset(class_images: dict, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=RANDOM_SEED):
    """
    Creates stratified reproducible splits for each class.
    """
    np.random.seed(seed)
    splits = {"train": {}, "val": {}, "test": {}}

    for cname, files in class_images.items():
        shuffled = list(files)
        np.random.shuffle(shuffled)
        n = len(shuffled)

        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        # Ensure at least 1 sample in validation and test if class has at least 3 samples
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n - n_train - n_val)
        else:
            n_test = n - n_train - n_val

        train_files = shuffled[:n_train]
        val_files = shuffled[n_train:n_train + n_val]
        test_files = shuffled[n_train + n_val:]

        splits["train"][cname] = train_files
        splits["val"][cname] = val_files
        splits["test"][cname] = test_files

    return splits


def plot_class_distribution(class_images: dict, output_path: Path):
    """
    Generates a class distribution bar chart.
    """
    classes = sorted(list(class_images.keys()))
    counts = [len(class_images[c]) for c in classes]

    # Clean display labels
    labels = [c.replace("___", "\n").replace("_", " ") for c in classes]

    plt.figure(figsize=(14, max(8, len(classes) * 0.35)))
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, counts, color="#2e7d32", edgecolor="#1b5e20", alpha=0.85)
    plt.yticks(y_pos, labels, fontsize=8)
    plt.xlabel("Number of Unique Images", fontsize=10, fontweight="bold")
    plt.title(f"PlantVillage Dataset - Class Distribution ({len(classes)} Classes, {sum(counts)} Total Images)", fontsize=12, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=200)
    plt.close()
    logger.info(f"Saved class distribution visualization to '{output_path}'.")


def main():
    parser = argparse.ArgumentParser(description="Prepare and split PlantVillage dataset.")
    parser.add_argument("--data-dir", type=str, default=str(DATASET_RAW_DIR), help="Path to raw dataset.")
    parser.add_argument("--output-dir", type=str, default=str(DATASET_SPLIT_DIR), help="Path to save splits manifest.")
    parser.add_argument("--artifacts-dir", type=str, default=str(ARTIFACTS_DIR), help="Directory to save class_names and charts.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    raw_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    art_dir = Path(args.artifacts_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preparing dataset from: {raw_dir}")
    logger.info(f"Output directory for manifest: {out_dir}")
    logger.info(f"Artifacts directory: {art_dir}")

    try:
        class_images, duplicate_count, corrupt_count, total_scanned = inspect_and_collect_images(raw_dir)
    except Exception as e:
        logger.error(f"Dataset preparation could not continue: {e}")
        sys.exit(1)

    classes = sorted(list(class_images.keys()))
    logger.info(f"Identified {len(classes)} non-empty classes.")

    # Save class_names.json into target artifacts directory
    class_names_target = art_dir / "class_names.json"
    with open(class_names_target, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)
    logger.info(f"Saved class mapping ({len(classes)} classes) to '{class_names_target}'.")

    # Create stratified splits
    splits = split_dataset(class_images, args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed)

    # Check for split coverage
    missing_classes = {"train": [], "val": [], "test": []}
    for split_name in ("train", "val", "test"):
        for c in classes:
            if len(splits[split_name].get(c, [])) == 0:
                missing_classes[split_name].append(c)

    # Construct rich splits manifest
    manifest_data = {
        "metadata": {
            "dataset_source": str(raw_dir),
            "random_seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "total_files_scanned": total_scanned,
            "total_usable_images": sum(len(v) for v in class_images.values()),
            "duplicates_pruned": duplicate_count,
            "corrupted_images_rejected": corrupt_count,
            "total_classes": len(classes)
        },
        "class_names": classes,
        "class_to_idx": {c: i for i, c in enumerate(classes)},
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"]
    }

    manifest_path = out_dir / "splits_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info(f"Saved splits manifest to '{manifest_path}'.")

    # Summary report
    summary = {
        "dataset_source": str(raw_dir),
        "random_seed": args.seed,
        "total_classes": len(classes),
        "total_scanned": total_scanned,
        "total_usable_images": sum(len(v) for v in class_images.values()),
        "duplicates_pruned": duplicate_count,
        "corrupt_images": corrupt_count,
        "train_images": sum(len(v) for v in splits["train"].values()),
        "val_images": sum(len(v) for v in splits["val"].values()),
        "test_images": sum(len(v) for v in splits["test"].values()),
        "missing_classes_in_splits": missing_classes,
        "per_class_counts": {
            c: {
                "total": len(class_images[c]),
                "train": len(splits["train"].get(c, [])),
                "val": len(splits["val"].get(c, [])),
                "test": len(splits["test"].get(c, []))
            }
            for c in classes
        }
    }
    summary_path = out_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved dataset summary to '{summary_path}'.")

    # Plot distribution chart into target artifacts directory
    plot_class_distribution(class_images, art_dir / "class_distribution.png")

    logger.info("Dataset preparation successfully completed.")
    logger.info(f"Summary: {summary['total_usable_images']} images partitioned into "
                f"{summary['train_images']} Train, {summary['val_images']} Val, {summary['test_images']} Test.")


if __name__ == "__main__":
    main()
