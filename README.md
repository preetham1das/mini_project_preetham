# AI-Based Forest Cover Segmentation and Temporal Deforestation Detection Using U-Net

An academic deep learning project for binary forest-cover semantic segmentation and post-processing temporal change detection on satellite imagery using a PyTorch U-Net architecture.

---

## 📌 Project Overview & Semantic Correctness

- **Semantic Task**: Binary Forest-Cover Segmentation (Stage 1) & Temporal Deforestation Detection (Stage 2).
- **Dataset Mask Mapping (Stage 1)**:
  - `Raw Mask Value 2` ➔ **Forest Cover** (Class 1)
  - `Raw Mask Value 1` ➔ **Background / Non-Forest** (Class 0)
- **Model Objective**: Learn pixel-level feature representations to separate **Forest** from **Non-Forest** (Stage 1). Learn temporal features to identify Deforestation (Stage 2).

### Two-Stage Analysis

1. **Stage 1 - Forest / Non-Forest Semantic Segmentation**:
The Amazon dataset provides forest-cover segmentation labels. We use a U-Net to perform forest/non-forest segmentation.

2. **Stage 2 - Temporal Forest-Cover Change Detection**:
Initially, we attempted to use the Stage 1 model to predict masks on T1 and T2 separately and compared them. However, due to the severe domain shift between the Amazon Forest Dataset and the new Forest-Change dataset, this approach produced poor results.
To address this, Stage 2 now implements a separate, dedicated 6-channel Temporal U-Net model trained directly on the Forest-Change dataset. It concatenates the T1 and T2 images and directly predicts a binary deforestation mask.
The Stage 2 Forest-Change dataset uses: 270 training, 31 validation, and 33 test pairs.

This project reports pixel counts and percentages. It does not report physical area such as hectares because reliable geospatial metadata and pixel resolution are not available in the current dataset. The generalization claims are strictly limited to the test splits of the respective datasets.

---

## 📁 Repository Structure

```
Deforestation-Detection-from-Satellite-Images/
│
├── data/
│   ├── Amazon Forest Dataset/
│   │   ├── Training/       (30 TIFF images + 30 PNG masks)
│   │   ├── Validation/     (15 TIFF images + 15 PNG masks)
│   │   └── Test/           (15 unlabeled TIFF images)
│   └── Forest-Change/extracted/Forest-Change-dataset/images/
│       ├── train/          (270 pairs)
│       ├── val/            (31 pairs)
│       └── test/           (33 pairs)
│
├── src/
│   ├── dataset.py          # PyTorch ForestDataset & DataLoader creation (Stage 1)
│   ├── temporal_dataset.py # Temporal dataset handling (Stage 2)
│   ├── model.py            # U-Net deep learning network architecture (Stage 1)
│   ├── temporal_model.py   # 6-Channel Temporal U-Net (Stage 2)
│   ├── unet.py             # Backwards compatibility import layer
│   ├── metrics.py          # BCE + Dice Loss & evaluation metrics (IoU, Dice, Acc)
│   ├── train.py            # Model training & checkpointing pipeline (Stage 1)
│   ├── train_temporal.py   # Model training & checkpointing pipeline (Stage 2)
│   ├── evaluate.py         # Validation dataset evaluation (Stage 1)
│   ├── evaluate_temporal.py# Test set evaluation (Stage 2)
│   ├── predict.py          # Test set inference & green overlay visualizer generator (Stage 1)
│   ├── deforestation_detection.py  # Temporal change detection script
│   └── visualize_temporal.py       # Stage 2 results visualization
│
├── models/
│   ├── best_unet_model.pth # Generated best model weights checkpoint (Stage 1)
│   └── best_temporal_model.pth # Generated best temporal model weights (Stage 2)
│
├── outputs/
│   ├── temporal/           # Temporal outputs and visualizations (Stage 2)
│   └── predictions/        # Stage 1 test predictions
│
└── README.md
```

---

## ⚙️ Environment Setup & Prerequisites

- **Python**: `3.14.6`
- **PyTorch**: `2.14.0+cpu`
- **torchvision**: `0.29.0`
- **NumPy**: `2.5.2` / `2.5.3`
- **OpenCV**: `5.0.0`
- **Matplotlib**: `3.11.1`
- **Pandas**: `3.0.5`
- **Pillow**: `12.3.0`

### Activate Virtual Environment
On Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

---

## 🚀 Execution Guide

### 1. Test Dataset Loader & Model Architecture (Stage 1 & 2)
```powershell
.\venv\Scripts\python.exe src/dataset.py
.\venv\Scripts\python.exe src/model.py
.\venv\Scripts\python.exe src/metrics.py
.\venv\Scripts\python.exe src/temporal_dataset.py
.\venv\Scripts\python.exe src/temporal_model.py
```

### 2. Train the U-Net Model (Stage 1)
Train the Stage 1 network on CPU:
```powershell
.\venv\Scripts\python.exe src/train.py
```

### 3. Train the Temporal Model (Stage 2)
Train the Stage 2 6-channel model on the Forest-Change dataset:
```powershell
.\venv\Scripts\python.exe src/train_temporal.py
```
*Outputs: `models/best_temporal_model.pth`, `outputs/temporal_training_history.csv`, `outputs/temporal_training_curves.png`*

### 4. Evaluate Validation Performance (Stage 1)
```powershell
.\venv\Scripts\python.exe src/evaluate.py
```

### 5. Evaluate Temporal Deforestation Performance (Stage 2)
Evaluate the Stage 2 model against the held-out test set:
```powershell
.\venv\Scripts\python.exe src/evaluate_temporal.py
```
*Outputs: `outputs/temporal/temporal_test_results.json`, `outputs/temporal/temporal_test_report.txt`*

### 6. Visualize Temporal Representative Results (Stage 2)
```powershell
.\venv\Scripts\python.exe src/visualize_temporal.py
```
*Outputs: Visualizations saved to `outputs/temporal/visualizations/`*

---

## 📊 Temporal Change Categories

| Category | Condition ($T1 \rightarrow T2$) | Color on Change Map |
| :--- | :--- | :--- |
| **Forest Loss (Deforestation)** | $\text{Forest } (1) \rightarrow \text{Non-Forest } (0)$ | 🔴 Red |
| **Forest Gain (Regrowth)** | $\text{Non-Forest } (0) \rightarrow \text{Forest } (1)$ | 🔵 Blue |
| **Stable Forest** | $\text{Forest } (1) \rightarrow \text{Forest } (1)$ | 🟢 Green |
| **Stable Non-Forest** | $\text{Non-Forest } (0) \rightarrow \text{Non-Forest } (0)$ | 🔘 Dark Gray |

---

## 📈 Final Model Performance Summary

### Stage 1 (Validation Metrics)
- **IoU**: `83.80%`
- **Dice**: `91.19%`
- **Precision**: `92.26%`
- **Recall**: `90.14%`
- **Pixel Accuracy**: `90.91%`

### Stage 2 (Test Metrics)
- **IoU**: `29.83%`
- **Dice**: `45.95%`
- **Precision**: `41.31%`
- **Recall**: `51.76%`

*Stage 2 was trained on Forest-Change Dataset (270 training, 31 validation, 33 test pairs).*

### Stage 2 Representative Visualizations (Test Set)
- **Best**: `test_000007` (IoU: 0.6142, Dice: 0.7610)
- **Median**: `test_000025` (IoU: 0.1089, Dice: 0.1965)
- **Worst**: `test_000014` (IoU: 0.0000, Dice: 0.0000)
