# Chest X-Ray Pneumonia Detection

Deep learning system for automated pneumonia detection from chest X-ray images using ensemble of TorchXRayVision DenseNet and ResNet50.

## Features

- **Advanced Preprocessing**: Text removal, lung segmentation, CLAHE enhancement
- **Ensemble Models**: TXV DenseNet121 + ResNet50 with Test-Time Augmentation
- **Interpretability**: Grad-CAM visualization for explainable predictions
- **Production Ready**: Optimized threshold, comprehensive evaluation metrics

## Requirements

```bash
pip pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Project Structure

```
project/
├── config.yaml              # Configuration file
├── preprocess.py           # Data preprocessing
├── train.py                # Model training
├── evaluate.py             # Model evaluation
├── predict.py              # Inference on new images
├── gradcam.py             # Grad-CAM visualization
├── data/                   # Raw dataset
│   ├── train/
│   │   ├── 0_Normal/
│   │   └── 1_Pneumonia/
│   ├── val/
│   └── test/
└── outputs/               # Generated outputs
    ├── models/            # Trained weights
    ├── evaluation/        # Metrics and plots
    ├── predict/          # Prediction results
    ├── gradcam/          # Heatmaps
    └── logs/             # Training logs
```

## Quick Start

### 1. Prepare Dataset

Organize your chest X-ray images:
```
data/
├── train/
│   ├── 0_Normal/
│   └── 1_Pneumonia/
├── val/
│   ├── 0_Normal/
│   └── 1_Pneumonia/
└── test/
    ├── 0_Normal/
    └── 1_Pneumonia/
```

### 2. Preprocess Images

```bash
python preprocess.py
```

This creates `data_masked/` with enhanced images ready for training.

### 3. Train Models

Train both models (recommended):
```bash
python train.py --model all
```

Train specific model:
```bash
python train.py --model txv_densenet
python train.py --model resnet50
```

### 4. Evaluate Performance

```bash
python evaluate.py
```

Outputs:
- `outputs/evaluation/confusion_matrix.png`
- `outputs/evaluation/roc_curve.png`
- `outputs/evaluation/report.txt`
- Optimal threshold saved to `config.yaml`

### 5. Make Predictions

Single image:
```bash
python predict.py path/to/image.jpg
```

Batch processing:
```bash
python predict.py path/to/folder/
```

Results saved to `outputs/predict/pred_<folder_name>.csv`

### 6. Generate Grad-CAM

```bash
python gradcam.py --image path/to/image.jpg --arch txv_densenet
python gradcam.py --image path/to/image.jpg --arch resnet50
```

Heatmaps saved to `outputs/gradcam/`

# Model paths
model_txv_densenet: outputs/models/txv_densenet_best.pth
model_resnet: outputs/models/resnet50_best.pth

# Inference
optimal_threshold: 0.5  # Auto-updated after evaluation
```

## Model Architecture

### TXV DenseNet121
- Pretrained on CheXpert, MIMIC-CXR, PadChest
- Grayscale input (1 channel)
- Custom classifier: Dropout(0.5) + Linear(1024 → 2)

### ResNet50
- Pretrained on ImageNet
- RGB input (3 channels)  
- Custom classifier: Dropout(0.5) + Linear(2048 → 2)

### Ensemble Strategy
- Average probabilities from both models
- Horizontal flip Test-Time Augmentation
- Threshold-based classification (optimized via F1-score)

## Performance Metrics

After training and evaluation:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curve
- Confusion matrix
- Per-class metrics

## Preprocessing Pipeline

1. **Text Removal**: Detects and inpaints border annotations
2. **Lung Segmentation**: Otsu thresholding + morphological operations
3. **Enhancement**: CLAHE, gamma correction, denoising
4. **Sharpening**: Unsharp masking for edge enhancement
5. **Normalization**: Model-specific normalization

## Training Features

- Focal Loss for class imbalance
- AdamW optimizer with weight decay
- Cosine annealing learning rate schedule
- Data augmentation: flip, rotation, affine transforms
- Early stopping (best validation accuracy)
- Comprehensive logging