"""
Sample Dataset Downloader for PlantVillage.
Downloads a lightweight, genuine sample subset of authentic PlantVillage leaf images
directly from the official GitHub repository (spMohanty/PlantVillage-Dataset)
for quick testing, rapid pipeline verification, and student demonstration.
"""

import sys
import argparse
import logging
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import DATASET_RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_API_URL = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"

# Representative classes for fast demonstration
DEFAULT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]


def download_sample_images(output_dir: Path, images_per_class: int = 10, classes=None):
    """
    Fetches real PlantVillage images from official GitHub repo.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target_classes = classes if classes else DEFAULT_CLASSES

    logger.info(f"Downloading authentic PlantVillage sample ({images_per_class} images/class) into: {output_dir}")

    total_downloaded = 0
    headers = {"User-Agent": "CropDiseaseDetection-AcademicProject"}

    for idx, cname in enumerate(target_classes, 1):
        class_folder = output_dir / cname
        class_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{idx}/{len(target_classes)}] Fetching directory list for: {cname}...")
        api_url = f"{REPO_API_URL}/{cname}"
        try:
            resp = requests.get(api_url, headers=headers, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch file list for {cname}: Status {resp.status_code}")
                continue

            files_info = resp.json()
            image_files = [f for f in files_info if f.get("name", "").lower().endswith((".jpg", ".jpeg", ".png"))]

            selected = image_files[:images_per_class]
            for file_info in selected:
                fname = file_info["name"]
                furl = file_info["download_url"]
                dest_path = class_folder / fname

                if dest_path.exists() and dest_path.stat().st_size > 0:
                    continue  # already downloaded

                img_resp = requests.get(furl, headers=headers, timeout=20)
                if img_resp.status_code == 200:
                    with open(dest_path, "wb") as f:
                        f.write(img_resp.content)
                    total_downloaded += 1
                else:
                    logger.warning(f"Failed to download image {fname}: {img_resp.status_code}")

        except Exception as e:
            logger.error(f"Error downloading class {cname}: {e}")

    logger.info(f"Sample download complete. Downloaded {total_downloaded} real leaf images.")


def main():
    parser = argparse.ArgumentParser(description="Download sample PlantVillage dataset from GitHub.")
    parser.add_argument("--output-dir", type=str, default=str(DATASET_RAW_DIR), help="Target directory.")
    parser.add_argument("--images-per-class", type=int, default=10, help="Number of images per class.")
    parser.add_argument("--all-classes", action="store_true", help="Download from all 38 classes instead of default 12.")
    args = parser.parse_args()

    classes = None
    if args.all_classes:
        # Fetch full class list from GitHub API
        try:
            r = requests.get(REPO_API_URL, headers={"User-Agent": "AcademicProject"}, timeout=15)
            if r.status_code == 200:
                classes = [item["name"] for item in r.json() if item.get("type") == "dir"]
        except Exception as e:
            logger.warning(f"Could not fetch full class list, using default classes: {e}")

    download_sample_images(Path(args.output_dir), args.images_per_class, classes=classes)


if __name__ == "__main__":
    main()
