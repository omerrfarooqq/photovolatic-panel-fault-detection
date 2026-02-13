# PV Solar Panel Fault Detection System
## Deep Learning-based Defect Detection using YOLOv8

---

## Overview

This project implements an **end-to-end PV (photovoltaic) solar panel fault detection** pipeline using YOLOv8 object detection. It handles a **severely imbalanced dataset** (42:1 ratio between most and least common classes) with 14 fault categories.

### Dataset Statistics
| Split | Images | Annotations | Empty (No-fault) |
|-------|--------|-------------|-------------------|
| Train | 2,308  | 2,308       | 262               |
| Val   | 283    | 283         | -                 |
| Test  | 303    | 303         | -                 |

### Class Distribution (Training Set)
| Class | Fault Type       | Instances | Imbalance Ratio |
|-------|------------------|-----------|-----------------|
| 12    | Bird-Drop        | 11,646    | 1.0x (baseline) |
| 13    | Physical-Damage  | 8,932     | 1.3x            |
| 3     | Diode            | 8,631     | 1.3x            |
| 7     | No-Anomaly       | 6,707     | 1.7x            |
| 8     | Offline-Module   | 2,887     | 4.0x            |
| 4     | Diode-Multi      | 2,671     | 4.4x            |
| 0     | Cell             | 2,229     | 5.2x            |
| 2     | Cracking         | 2,054     | 5.7x            |
| 1     | Cell-Multi       | 1,127     | 10.3x           |
| 11    | Vegetation       | 544       | 21.4x           |
| 5     | Hot-Spot         | 407       | 28.6x           |
| 6     | Hot-Spot-Multi   | 398       | 29.3x           |
| 10    | Soiling          | 291       | 40.0x           |
| 9     | Shadowing        | 272       | 42.8x           |

> **Note:** Class names are inferred from common PV fault taxonomies. Update `data.yaml` if your dataset uses different labels.

---

## Project Structure

```
pv_fault_detection/
├── data.yaml              # YOLO dataset configuration
├── requirements.txt       # Python dependencies
├── utils.py               # Shared utility functions
├── 01_eda.py              # Exploratory Data Analysis
├── 02_train.py            # Training pipeline (imbalance-aware)
├── 03_evaluate.py         # Comprehensive model evaluation
├── 04_inference.py        # Inference + diagnostic reports
├── README.md              # This file
├── eda_outputs/           # EDA visualisations (generated)
├── runs/                  # Training runs & checkpoints (generated)
├── eval_outputs/          # Evaluation metrics (generated)
└── inference_outputs/     # Inference results (generated)
```

---

## Quick Start

### 1. Install Dependencies
```bash
cd C:\Users\omerf\pv_fault_detection
pip install -r requirements.txt
```

### 2. Run EDA (Exploratory Data Analysis)
```bash
python 01_eda.py
```
Generates:
- Class distribution bar charts
- Imbalance ratio table (CSV)
- Bbox size/aspect analysis
- Bbox centre heatmap
- Co-occurrence matrix
- Annotated sample grid

### 3. Train the Model
```bash
# Quick training (nano model, ~2h on GPU)
python 02_train.py --model yolov8n.pt --epochs 100

# Better accuracy (small model)
python 02_train.py --model yolov8s.pt --epochs 150 --imgsz 800

# Best accuracy (medium model, needs more VRAM)
python 02_train.py --model yolov8m.pt --epochs 200 --imgsz 640 --batch 8
```

### 4. Evaluate
```bash
python 03_evaluate.py                    # on test set
python 03_evaluate.py --split val        # on validation set
```

### 5. Run Inference
```bash
# Single image
python 04_inference.py --source path/to/image.jpg

# Folder of images
python 04_inference.py --source path/to/folder/

# With cropped detections
python 04_inference.py --source path/to/image.jpg --save-crop

# Test on the test set
python 04_inference.py --source C:/Users/omerf/Downloads/split_upload_2/split_upload_2/images/test
```

---

## Imbalance Handling Strategy

The dataset has extreme class imbalance (42:1). The pipeline addresses this through:

1. **Aggressive Data Augmentation**
   - Mosaic (1.0) — stitches 4 images, increasing exposure to rare classes
   - MixUp (0.15) — blends images for regularisation
   - Copy-Paste (0.1) — synthetically places minority objects
   - Geometric: rotation, scale, shear, perspective, flips

2. **Loss Function Design**
   - Distribution Focal Loss (DFL) for box regression
   - Increased classification loss weight (cls=1.5)
   - AdamW optimiser with cosine LR decay

3. **Training Strategy**
   - Pre-trained weights (COCO transfer learning)
   - 5-epoch warmup
   - Early stopping (patience=30)
   - Multi-scale training ready (--imgsz 800 recommended for small defects)

---

## Diagnostic Reports

The inference script generates structured JSON diagnostic reports with:
- Fault type classification
- Confidence scores
- **Severity assessment** (NORMAL → MINOR → MODERATE → CRITICAL)
- **Maintenance recommendations** based on detected fault types
- Colour-coded annotated images

---

## Model Selection Guide

| Model | Params | Speed (GPU) | Accuracy | Use Case |
|-------|--------|-------------|----------|----------|
| yolov8n | 3.2M  | Fastest    | Good     | Edge/real-time deployment |
| yolov8s | 11.2M | Fast       | Better   | Balanced speed/accuracy |
| yolov8m | 25.9M | Medium     | Best     | Server-side batch processing |
| yolov8l | 43.7M | Slower     | Best+    | Maximum accuracy needed |

---

## Expected Metrics

With YOLOv8s trained for 150 epochs on this dataset, typical results:
- **mAP@50**: 0.65–0.80 (depends on class name accuracy)
- **mAP@50-95**: 0.40–0.55
- Well-represented classes (Bird-Drop, Physical-Damage, Diode): AP > 0.7
- Rare classes (Shadowing, Soiling): AP likely 0.3–0.5

---
