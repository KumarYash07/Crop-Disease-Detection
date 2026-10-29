# Model Artifacts Directory

This directory stores all trained models, class metadata, training histories, and evaluation charts generated during the training and validation pipelines.

---

## Expected Artifacts

When model training and evaluation are executed, the following files will be produced:

| Artifact File | Description | Generating Script |
| :--- | :--- | :--- |
| `crop_disease_model.keras` | Serialized Keras/TensorFlow CNN model weights and architecture | `training/train_model.py` |
| `class_names.json` | Ordered list of verified class labels matching model output indices | `training/prepare_dataset.py` / `train_model.py` |
| `training_history.json` | Per-epoch loss and accuracy values (training and validation) | `training/train_model.py` |
| `evaluation_metrics.json` | Comprehensive test set metrics (Test Accuracy, Loss, Precision, Recall, F1) | `training/evaluate_model.py` |
| `training_curves.png` | Matplotlib plot comparing Training vs Validation Loss & Accuracy curves | `training/evaluate_model.py` |
| `confusion_matrix.png` | Normalized confusion matrix visualization | `training/evaluate_model.py` |

---

## Academic Integrity Notice

No placeholder or fake metric files are pre-committed. These artifacts are generated strictly when actual dataset preparation, training, and evaluation scripts execute on real data.
