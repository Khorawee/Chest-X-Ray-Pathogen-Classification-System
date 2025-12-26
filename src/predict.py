import os
import sys
import yaml
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms

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
except FileNotFoundError:
    print("Warning: config.yaml not found! Using defaults...")
    CONFIG = {}
except Exception as e:
    print(f"Error loading config: {e}")
    CONFIG = {}

CLASSES = CONFIG.get("classes", ["0_Normal", "1_Pneumonia"])
IMG_SIZE = CONFIG.get("image_size", 224)
THRESHOLD = CONFIG.get("optimal_threshold", 0.5)

MODEL_DN = CONFIG.get("model_txv_densenet", "outputs/models/txv_densenet_best.pth")
MODEL_RN = CONFIG.get("model_resnet", "outputs/models/resnet50_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# Define transforms with enhanced preprocessing
transform_gray = transforms.Compose([
    Enhanced_Preprocessing(),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

transform_rgb = transforms.Compose([
    Enhanced_Preprocessing(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_models():
    """Load trained models for inference"""
    print("\nLoading models...")
    models_loaded = []

    # Load DenseNet (TorchXRayVision)
    if XRV_AVAILABLE and os.path.exists(MODEL_DN):
        try:
            m_dn = xrv.models.DenseNet(weights="densenet121-res224-all")
            num_ftrs = m_dn.classifier.in_features
            
            # Match training architecture
            m_dn.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, len(CLASSES))
            )
            
            if hasattr(m_dn, "op_threshs"):
                m_dn.op_threshs = None

            m_dn.load_state_dict(torch.load(MODEL_DN, map_location=DEVICE))
            m_dn.to(DEVICE).eval()
            models_loaded.append(("txv_densenet", m_dn))
            print("   TXV DenseNet loaded")
        except Exception as e:
            print(f"   Failed to load DenseNet: {e}")

    # Load ResNet50
    if os.path.exists(MODEL_RN):
        try:
            m_rn = models.resnet50(weights=None)
            num_ftrs = m_rn.fc.in_features
            
            # Match training architecture
            m_rn.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, len(CLASSES))
            )

            m_rn.load_state_dict(torch.load(MODEL_RN, map_location=DEVICE))
            m_rn.to(DEVICE).eval()
            models_loaded.append(("resnet50", m_rn))
            print("   ResNet50 loaded")
        except Exception as e:
            print(f"   Failed to load ResNet50: {e}")

    if not models_loaded:
        raise RuntimeError("No model loaded. Check model paths in config.yaml")

    return models_loaded


@torch.no_grad()
def predict_image(img_path, models_loaded):
    """Predict class and confidence for a single image"""
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image: {e}")

    probs = []

    for name, model in models_loaded:
        try:
            if name == "txv_densenet":
                x = transform_gray(image).unsqueeze(0).to(DEVICE)
            else:
                x = transform_rgb(image).unsqueeze(0).to(DEVICE)
            
            # TTA: horizontal flip
            xf = torch.flip(x, dims=[3])

            p1 = torch.softmax(model(x), dim=1)
            p2 = torch.softmax(model(xf), dim=1)
            probs.append((p1 + p2) / 2)
            
        except Exception as e:
            print(f"   Model {name} failed: {e}")
            continue

    if not probs:
        raise RuntimeError("All models failed to predict")

    # Calculate ensemble average
    final_prob = torch.mean(torch.stack(probs), dim=0)[0, 1].item()

    pred_class = CLASSES[1] if final_prob >= THRESHOLD else CLASSES[0]
    confidence = final_prob if final_prob >= THRESHOLD else 1 - final_prob

    return pred_class, confidence, final_prob


def process_folder(folder):
    """Process all images in a folder and save results to CSV"""
    models_loaded = load_models()
    results = []

    files = [f for f in os.listdir(folder)
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]

    if not files:
        print(f"No images found in {folder}")
        return

    print(f"\nProcessing {len(files)} images | Threshold={THRESHOLD:.2f}")

    for f in tqdm(files, ncols=100):
        img_path = os.path.join(folder, f)
        try:
            cls, conf, prob = predict_image(img_path, models_loaded)
            results.append({
                "Image": f,
                "Prediction": cls,
                "Confidence": f"{conf:.4f}",
                "Pneumonia_Prob": f"{prob:.4f}",
            })
        except Exception as e:
            print(f"\nSkip {f}: {e}")
            results.append({
                "Image": f,
                "Prediction": "ERROR",
                "Confidence": "N/A",
                "Pneumonia_Prob": "N/A",
            })

    # Save results to CSV
    os.makedirs("outputs/predict", exist_ok=True)
    out_csv = f"outputs/predict/pred_{os.path.basename(folder)}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_or_folder>")
        sys.exit(1)

    path = sys.argv[1]

    if not os.path.exists(path):
        print(f"Error: Path not found: {path}")
        sys.exit(1)

    models_loaded = load_models()

    if os.path.isdir(path):
        process_folder(path)
    else:
        try:
            cls, conf, prob = predict_image(path, models_loaded)
            print("\nRESULT")
            print(f"Class      : {cls}")
            print(f"Confidence : {conf*100:.2f}%")
            print(f"Probability: {prob*100:.2f}%")
        except Exception as e:
            print(f"\nPrediction failed: {e}")
            sys.exit(1)