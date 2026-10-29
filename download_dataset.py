from huggingface_hub import hf_hub_download
from pathlib import Path
import zipfile

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

print("Downloading the original PlantVillage image archive...")

zip_path = hf_hub_download(
    repo_id="mohanty/PlantVillage",
    repo_type="dataset",
    filename="data.zip",
)

print("Archive downloaded:", zip_path)
print("Extracting images...")

extract_dir = output_dir / "PlantVillage_Images"
extract_dir.mkdir(exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as archive:
    archive.extractall(extract_dir)

print("Extraction completed:", extract_dir.resolve())