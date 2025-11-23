# CornViT

A Multi-Stage Convolutional Vision Transformer Framework for Corn Kernel Analysis

## Overview

Three-stage hierarchical classification pipeline for automated corn kernel quality assessment:

- **Stage 1**: Purity detection (Pure vs Impure)
- **Stage 2**: Shape classification (Flat vs Round)
- **Stage 3**: Embryo orientation (Up vs Down)

## Architecture

- **Model**: CvT-13 (384×384) with ImageNet-22k pretraining
- **Framework**: PyTorch + Microsoft CvT
- **Test Accuracy**: 93.8% (Stage 1), 94.1% (Stage 2), 91.1% (Stage 3)

## Setup

```bash
# Clone repository
git clone https://github.com/microsoft/CvT.git

# Install dependencies
pip install -r requirements.txt
```

## Training

Each stage has independent training scripts:

```bash
python stage1/train_cvt13.py  # Purity classification
python stage2/train_cvt13.py  # Shape classification
python stage3/train_cvt13.py  # Embryo orientation
```

## Inference

```bash
python stage1/inference_cvt13.py
python stage2/inference_cvt13.py
python stage3/inference_cvt13.py
```

## Baselines

ResNet50 and DenseNet121 baselines available in `baselines/`.

## Structure

```
├── stage1/          # Purity classification
├── stage2/          # Shape classification
├── stage3/          # Embryo orientation
├── baselines/       # Baseline models
├── CvT/             # Microsoft CvT implementation
├── preprocess/      # Data preprocessing scripts
└── raw_data/        # Raw corn kernel images
```

## Requirements

- Python 3.13+
- PyTorch 2.9+
- CUDA (optional, for GPU training)