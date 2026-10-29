# Dataset Directory

This directory is intended to hold the **PlantVillage** dataset used for training, validating, and testing the Convolutional Neural Network (CNN) model for crop disease detection.

---

## 1. About the PlantVillage Dataset

The PlantVillage dataset is an open-access repository of plant leaf images originally published by David P. Hughes and Marcel Salathé (2015), curated by Penn State University and EPFL.

- **Official Public Repository**: [https://github.com/spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- **Total Classes**: 38 classes (across 14 crop species: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato)
- **Total Images**: ~54,303 high-resolution RGB images
- **Image Size**: ~256x256 pixels
- **Download Size**: ~800 MB (compressed) / ~1.5 GB to 2.0 GB (uncompressed)

---

## 2. Recommended Directory Structure

Extract the dataset so that each class folder resides under `data/PlantVillage/`:

```
CropDiseaseDetection/
└── data/
    └── PlantVillage/
        ├── Apple___Apple_scab/
        ├── Apple___Black_rot/
        ├── Apple___Cedar_apple_rust/
        ├── Apple___healthy/
        ├── Blueberry___healthy/
        ├── Cherry_(including_sour)___Powdery_mildew/
        ├── Cherry_(including_sour)___healthy/
        ├── Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
        ├── Corn_(maize)___Common_rust_/
        ├── Corn_(maize)___Northern_Leaf_Blight/
        ├── Corn_(maize)___healthy/
        ├── Grape___Black_rot/
        ├── Grape___Esca_(Black_Measles)/
        ├── Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/
        ├── Grape___healthy/
        ├── Orange___Haunglongbing_(Citrus_greening)/
        ├── Peach___Bacterial_spot/
        ├── Peach___healthy/
        ├── Pepper,_bell___Bacterial_spot/
        ├── Pepper,_bell___healthy/
        ├── Potato___Early_blight/
        ├── Potato___Late_blight/
        ├── Potato___healthy/
        ├── Raspberry___healthy/
        ├── Soybean___healthy/
        ├── Squash___Powdery_mildew/
        ├── Strawberry___Leaf_scorch/
        ├── Strawberry___healthy/
        ├── Tomato___Bacterial_spot/
        ├── Tomato___Early_blight/
        ├── Tomato___Late_blight/
        ├── Tomato___Leaf_Mold/
        ├── Tomato___Septoria_leaf_spot/
        ├── Tomato___Spider_mites Two-spotted_spider_mite/
        ├── Tomato___Target_Spot/
        ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
        ├── Tomato___Tomato_mosaic_virus/
        └── Tomato___healthy/
```

---

## 3. How to Download the Dataset

### Option A: Clone from GitHub
```bash
git clone https://github.com/spMohanty/PlantVillage-Dataset.git
# Copy raw/color folder contents into data/PlantVillage/
```

### Option B: Kaggle / Direct Archive
You can also download the pre-organized dataset from Kaggle:
```bash
kaggle datasets download -d emmarex/plantdisease
# Unzip contents into data/PlantVillage/
```

---

## 4. Dataset Preparation

Once the dataset folder is placed, run the dataset preparation script:
```powershell
python training/prepare_dataset.py --data-dir ./data/PlantVillage --output-dir ./data/processed
```

This will:
1. Verify image integrity and valid file extensions (`.jpg`, `.jpeg`, `.png`).
2. Detect and handle duplicates or corrupted images.
3. Generate stratified 70% Train / 15% Validation / 15% Test splits with fixed random seed (42).
4. Save the verified class mapping to `artifacts/class_names.json`.
5. Generate a class distribution summary and chart.

---

## 5. Important Scientific Limitation

> **Note on Environmental Bias**:
> The PlantVillage dataset was primarily captured in controlled laboratory settings with uniform neutral backgrounds (light gray/black boards). Consequently, models trained exclusively on PlantVillage may experience performance degradation when deployed in real-world agricultural fields under complex backgrounds, varied sunlight, shadows, moisture, and dual-leaf overlap. This is a recognized limitation in computer vision for precision agriculture.
