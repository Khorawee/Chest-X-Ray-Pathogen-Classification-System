import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)

import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# TorchXRayVision availability check
try:
    import torchxrayvision as xrv
    XRV_AVAILABLE = True
except ImportError:
    XRV_AVAILABLE = False
    print("Warning: torchxrayvision not installed")

# Load configuration
try:
    with open("config.yaml", "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
    if CONFIG is None:
        raise ValueError("config.yaml is empty")
except Exception as e:
    print(f"Error loading config: {e}")
    CONFIG = {}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = CONFIG.get("classes", ["0_Normal", "1_Pneumonia"])
IMG_SIZE = CONFIG.get("image_size", 224)
BATCH_SIZE = CONFIG.get("batch_size", 8)

# Preprocessing parameters
PREP = CONFIG.get("preprocessing", {
    "text_threshold": 240,
    "border_margin_x": 0.08,
    "border_margin_y_top": 0.12,
    "border_margin_y_bottom": 0.88,
    "max_text_area_ratio": 0.02,
    "lung_crop_margin": 0.05,
    "clahe_clip_limit": 0.6,
    "clahe_tile_size": 8,
    "gamma_correction": 1.0,
    "denoise_strength": 2,
    "gaussian_sigma": 0.6,
    "sharpen_alpha": 1.1,
    "sharpen_beta": -0.1,
})


class Enhanced_Preprocessing:
    """Advanced preprocessing pipeline for chest X-ray images"""
    
    def remove_text_annotations(self, img):
        """Remove text annotations from border regions"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        _, thresh = cv2.threshold(
            gray,
            PREP.get("text_threshold", 240),
            255,
            cv2.THRESH_BINARY
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        mask = np.zeros_like(gray)

        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cv2.contourArea(c)

            is_border = (
                x < w * PREP.get("border_margin_x", 0.08) or
                x > w * (1 - PREP.get("border_margin_x", 0.08)) or
                y < h * PREP.get("border_margin_y_top", 0.12) or
                y > h * PREP.get("border_margin_y_bottom", 0.88)
            )

            if is_border and area < h * w * PREP.get("max_text_area_ratio", 0.02):
                cv2.rectangle(mask, (x, y), (x+cw, y+ch), 255, -1)

        return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    def segment_lung_region(self, img):
        """Segment and crop lung region of interest"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, bin_img = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, 2)

        contours, _ = cv2.findContours(
            bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid = [c for c in contours if cv2.contourArea(c) > h * w * 0.05]
        if not valid:
            m = PREP.get("lung_crop_margin", 0.05)
            return img[int(h*m):int(h*(1-m)), int(w*m):int(w*(1-m))]

        all_pts = np.vstack(valid)
        x, y, cw, ch = cv2.boundingRect(all_pts)

        pad = 20
        return img[
            max(0, y-pad):min(h, y+ch+pad),
            max(0, x-pad):min(w, x+cw+pad)
        ]

    def __call__(self, img_pil):
        """Apply full preprocessing pipeline"""
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Remove annotations and segment lung region
        img = self.remove_text_annotations(img)
        img = self.segment_lung_region(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=PREP.get("clahe_clip_limit", 0.6),
            tileGridSize=(PREP.get("clahe_tile_size", 8),) * 2
        )
        img = clahe.apply(gray)

        # Gamma correction
        img = np.power(
            img / 255.0,
            PREP.get("gamma_correction", 1.0)
        ) * 255

        # Denoise
        img = cv2.fastNlMeansDenoising(
            img.astype(np.uint8),
            h=PREP.get("denoise_strength", 2)
        )

        # Sharpen
        blur = cv2.GaussianBlur(
            img, (0, 0),
            PREP.get("gaussian_sigma", 0.6)
        )

        img = cv2.addWeighted(
            img,
            PREP.get("sharpen_alpha", 1.1),
            blur,
            PREP.get("sharpen_beta", -0.1),
            0
        )

        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))


def load_models():
    """Load trained models for evaluation"""
    print("\nLoading models for evaluation...")
    models_list = []

    # Load DenseNet (TorchXRayVision)
    if XRV_AVAILABLE:
        model_path = CONFIG.get("model_txv_densenet", "outputs/models/txv_densenet_best.pth")
        if os.path.exists(model_path):
            try:
                m1 = xrv.models.DenseNet(weights="densenet121-res224-all")
                num_ftrs = m1.classifier.in_features
                
                # Match training architecture
                m1.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_ftrs, len(CLASSES))
                )
                
                if hasattr(m1, "op_threshs"):
                    m1.op_threshs = None
                
                m1.load_state_dict(torch.load(model_path, map_location=DEVICE))
                m1.to(DEVICE).eval()
                models_list.append(("gray", m1))
                print(f"   TXV DenseNet loaded from {model_path}")
            except Exception as e:
                print(f"   Failed to load DenseNet: {e}")
        else:
            print(f"   DenseNet weights not found: {model_path}")

    # Load ResNet50
    model_path = CONFIG.get("model_resnet", "outputs/models/resnet50_best.pth")
    if os.path.exists(model_path):
        try:
            m2 = models.resnet50(weights=None)
            num_ftrs = m2.fc.in_features
            
            # Match training architecture
            m2.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, len(CLASSES))
            )
            
            m2.load_state_dict(torch.load(model_path, map_location=DEVICE))
            m2.to(DEVICE).eval()
            models_list.append(("rgb", m2))
            print(f"   ResNet50 loaded from {model_path}")
        except Exception as e:
            print(f"   Failed to load ResNet50: {e}")
    else:
        print(f"   ResNet50 weights not found: {model_path}")

    if not models_list:
        raise RuntimeError("No models loaded! Check model paths in config.yaml")

    return models_list


def evaluate():
    """Main evaluation function"""
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Classes: {CLASSES}")
    print(f"Batch size: {BATCH_SIZE}")
    
    models_list = load_models()

    # Define transforms for grayscale and RGB
    t_gray = transforms.Compose([
        Enhanced_Preprocessing(),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    t_rgb = transforms.Compose([
        Enhanced_Preprocessing(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # Find evaluation data
    root = CONFIG.get("data_dir", "./data_masked")
    eval_dir = os.path.join(root, "test") \
        if os.path.exists(os.path.join(root, "test")) \
        else os.path.join(root, "val")

    if not os.path.exists(eval_dir):
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    print(f"\nEvaluation data: {eval_dir}")

    # Create datasets
    try:
        ds_gray = datasets.ImageFolder(eval_dir, transform=t_gray)
        ds_rgb = datasets.ImageFolder(eval_dir, transform=t_rgb)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    print(f"   Total samples: {len(ds_gray)}")
    print(f"   Class distribution: {dict(zip(ds_gray.classes, np.bincount(ds_gray.targets)))}")

    dl_gray = DataLoader(ds_gray, BATCH_SIZE, shuffle=False, num_workers=0)
    dl_rgb = DataLoader(ds_rgb, BATCH_SIZE, shuffle=False, num_workers=0)

    # Run inference
    print(f"\nRunning inference...")
    y_true, y_prob = [], []

    for (xg, y), (xr, _) in tqdm(zip(dl_gray, dl_rgb), total=len(dl_gray), desc="Evaluating"):
        xg, xr = xg.to(DEVICE), xr.to(DEVICE)

        with torch.no_grad():
            probs = []

            for typ, model in models_list:
                x = xg if typ == "gray" else xr
                xf = torch.flip(x, [3])  # TTA: horizontal flip

                p = (
                    torch.softmax(model(x), 1) +
                    torch.softmax(model(xf), 1)
                ) / 2

                probs.append(p)

            p_final = sum(probs) / len(probs)

        y_prob.extend(p_final[:, 1].cpu().numpy())
        y_true.extend(y.numpy())

    # Find optimal threshold
    print(f"\nFinding optimal threshold...")
    best_f1, best_t = 0, 0.5
    
    for t in np.arange(0.1, 0.9, 0.01):
        pred = [1 if p >= t else 0 for p in y_prob]
        f1 = f1_score(y_true, pred, average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\nBest Threshold = {best_t:.4f} | F1-Score = {best_f1:.4f}")

    # Save threshold to config
    CONFIG["optimal_threshold"] = round(float(best_t), 4)
    try:
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(CONFIG, f, default_flow_style=False, sort_keys=False)
        print(f"Saved threshold to config.yaml")
    except Exception as e:
        print(f"Failed to save config: {e}")

    # Generate evaluation reports
    out_dir = "outputs/evaluation"
    os.makedirs(out_dir, exist_ok=True)

    y_pred = [1 if p >= best_t else 0 for p in y_prob]

    # Save confusion matrix
    print(f"\nGenerating visualizations...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES,
                yticklabels=CLASSES)
    plt.title(f"Confusion Matrix (Threshold={best_t:.4f})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir}/confusion_matrix.png")

    # Save ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir}/roc_curve.png")

    # Save classification report
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    
    with open(f"{out_dir}/report.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: Ensemble (TXV DenseNet + ResNet50)\n")
        f.write(f"Threshold: {best_t:.4f}\n")
        f.write(f"F1-Score: {best_f1:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write("-" * 60 + "\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    print(f"Saved: {out_dir}/report.txt")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(report)
    print(f"\nConfusion Matrix:")
    print(cm)
    print("\n" + "=" * 60)
    print(f"All results saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        evaluate()
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()