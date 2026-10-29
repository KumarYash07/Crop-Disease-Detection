"""
Standalone CLI Prediction Script for Crop Disease Detection.
Allows testing model inference directly from the terminal on an individual leaf image.
Usage:
    python training/predict_image.py --image path/to/leaf.jpg
"""

import sys
import argparse
import logging
from pathlib import Path

# Add root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from models.predictor import predict_crop_disease
from models.preprocessing import validate_image_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Predict crop disease from a single leaf photograph.")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image file.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top candidates to report.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image file '{image_path}' does not exist.")
        sys.exit(1)

    is_valid, reason = validate_image_file(str(image_path), filename=image_path.name)
    if not is_valid:
        logger.error(f"Image validation failed: {reason}")
        sys.exit(1)

    try:
        result = predict_crop_disease(str(image_path), top_k=args.top_k)
    except Exception as e:
        logger.error(f"Inference execution failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("      CROP DISEASE DETECTION - INFERENCE RESULT")
    print("=" * 60)
    print(f"Target Image:      {image_path.name}")
    print(f"Predicted Crop:    {result['crop']}")
    print(f"Diagnosis:         {result['disease']}")
    print(f"Health Status:     {'Healthy Foliage' if result['is_healthy'] else 'Disease Detected'}")
    print(f"Model Confidence:  {result['confidence_percentage']}%")
    print("-" * 60)
    print("Top Candidate Predictions:")
    for cand in result["top_predictions"]:
        print(f"  {cand['rank']}. {cand['display_name']} ({cand['confidence_percent']}%)")
    print("-" * 60)

    info = result["disease_information"]
    print(" agronomic overview:")
    print(f"  {info.get('description', 'N/A')}\n")

    print("Visible Symptoms:")
    for s in info.get("symptoms", []):
        print(f"  • {s}")
    print()

    print("Preventive Practices:")
    for p in info.get("prevention", []):
        print(f"  • {p}")
    print()

    print("Management Recommendations:")
    for m in info.get("management", []):
        print(f"  • {m}")
    print("=" * 60)
    print(f"\n{result['disclaimer']}\n")


if __name__ == "__main__":
    main()
