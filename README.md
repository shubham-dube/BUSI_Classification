# 🩺 BUSI Breast Ultrasound Image Classification

**Multi-class classification of breast ultrasound images into Normal, Benign, and Malignant categories.**
---

## 📋 Overview

This project implements a complete, clinically-responsible deep learning pipeline for classifying breast ultrasound scans. It addresses two core challenges in medical imaging:

- **Data leakage** — solved via patient-level hold-out splitting
- **Class imbalance** — solved via SMOTE on extracted feature vectors

**Architecture:** Frozen MobileNetV2 (feature extractor) + fully-connected classifier head trained with Focal Loss.

---

## 📊 Results at a Glance

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **80.89%** |
| Benign F1-Score | 0.85 |
| Malignant F1-Score | 0.76 |
| Normal F1-Score | 0.76 |
| Malignant AUC (ROC) | 0.949 |
| Normal AUC (ROC) | 0.960 |

---

## 📁 Dataset

**BUSI — Breast Ultrasound Images Dataset**
> W. Al-Dhabyani, M. Gomaa, H. Khaled, A. Fahmy. *Dataset of breast ultrasound images.* Data in Brief, 2020. [DOI: 10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863)

| Class | Images | Split |
|-------|--------|-------|
| Benign (B) | 437 | ~80/20 patient-level |
| Malignant (M) | 210 | ~80/20 patient-level |
| Normal (N) | 133 | ~80/20 patient-level |
| **Total** | **780** | Masks excluded |

**Expected directory layout:**
```
Dataset_BUSI_with_GT/
├── benign/
│   ├── benign (1).png
│   ├── benign (1)_mask.png
│   └── ...
├── malignant/
│   └── ...
└── normal/
    └── ...
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install tensorflow scikit-learn imbalanced-learn matplotlib seaborn pillow numpy pandas tqdm
```

### 2. Configure the dataset path
Open `BUSI_Classification_Pipeline.ipynb` and update:
```python
CONFIG = {
    'data_dir': 'Dataset_BUSI_with_GT',   # <-- your path here
    ...
}
```

### 3. Run the notebook
```bash
jupyter notebook BUSI_Classification_Pipeline.ipynb
```
Run all cells top to bottom. All figures and model files are saved automatically.

---

## 🏗️ Pipeline Architecture

```
Input (224×224×3)
        │
        ▼
MobileNetV2 (ImageNet, frozen)
        │  Global Average Pooling
        ▼
  Feature Vector (1280-dim)
        │
    [SMOTE applied here — train set only]
        │
        ▼
  Dense 512 → BatchNorm → ReLU → Dropout(0.4)
        │
  Dense 256 → BatchNorm → ReLU → Dropout(0.4)
        │
  Dense 128 → BatchNorm → ReLU → Dropout(0.2)
        │
        ▼
  Dense 3 → Softmax  [B / M / N]
```

**Loss function — Focal Loss:**
```
FL(pₜ) = −αₜ · (1 − pₜ)^γ · log(pₜ)
         γ = 2.0,  α = 0.25
```

---

## 🔑 Key Design Decisions

### Patient-Level Hold-out Split
Images from the same patient never appear in both train and test sets. A unique patient key `(class_folder, patient_id)` is used to avoid cross-class ID collisions (e.g. `benign (1)` and `malignant (1)` are different patients).

```python
# Zero-overlap validation
assert len(train_patient_keys & test_patient_keys) == 0
```

### SMOTE on Features (not pixels)
Applying SMOTE directly to 1280-dim feature vectors (not raw images) avoids generating visually unrealistic synthetic images while still perfectly balancing the three classes.

| Class | Before SMOTE | After SMOTE |
|-------|-------------|-------------|
| B | 1,047 | 1,047 |
| M | 504 | 1,047 |
| N | 318 | 1,047 |

### Focal Loss
Standard cross-entropy treats all mistakes equally. Focal Loss exponentially down-weights easy examples, forcing the network to focus training signal on hard cases like subtle malignant lesions.

---

## 📈 Output Figures

| File | Description |
|------|-------------|
| `class_distribution.png` | Dataset class counts |
| `augmented_samples.png` | Sample augmented training images |
| `smote_comparison.png` | Before vs after SMOTE |
| `training_history.png` | Focal loss & accuracy curves |
| `confusion_matrix.png` | Count + normalised confusion matrix |
| `per_class_metrics.png` | Precision / Recall / F1 per class |
| `roc_curves.png` | One-vs-Rest ROC curves |
| `prediction_grid.png` | Correct & misclassified test samples |

---

## 💾 Saved Artefacts

| File | Description |
|------|-------------|
| `busi_classifier_head.keras` | Trained classifier head |
| `best_classifier.keras` | Best checkpoint (by val accuracy) |
| `label_classes.npy` | Label encoder class array |
| `X_test_feats.npy` | Extracted test features |
| `y_test.npy` | Test ground truth labels |

---

## 🔮 Inference on a New Image

```python
result = predict_single_image(
    image_path   = "path/to/scan.png",
    feature_extractor = feature_extractor,
    classifier        = classifier,
    label_encoder     = le,
    img_size          = (224, 224)
)
# {'predicted_class': 'B', 'confidence': 0.91, 'all_probabilities': {'B': 0.91, 'M': 0.06, 'N': 0.03}}
```

---

## 🧰 Tech Stack

| Library | Purpose |
|---------|---------|
| TensorFlow / Keras | MobileNetV2, model training |
| scikit-learn | Metrics, label encoding |
| imbalanced-learn | SMOTE |
| Pillow / NumPy | Image loading & preprocessing |
| Matplotlib / Seaborn | Visualisation |

---

## 📄 Citation

```bibtex
@article{aldhabyani2020,
  title   = {Dataset of breast ultrasound images},
  author  = {Al-Dhabyani, Walid and Gomaa, Mohammed and Khaled, Hussien and Fahmy, Aly},
  journal = {Data in Brief},
  volume  = {28},
  year    = {2020},
  doi     = {10.1016/j.dib.2019.104863}
}
```

---

## 📬 License

This project is released under the [MIT License](LICENSE).