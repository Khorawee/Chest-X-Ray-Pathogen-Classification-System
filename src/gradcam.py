import argparse
import os
import sys
import yaml
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# Grad-CAM library
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    sys.exit("Error: Install Grad-CAM using 'pip install grad-cam'")

# TorchXRayVision availability check
try:
    import torchxrayvision as xrv
    XRV_AVAILABLE = True
except ImportError:
    XRV_AVAILABLE = False
    print("Warning: torchxrayvision not installed")

# Default preprocessing parameters
DEFAULT_PREP = {
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
}

# Load configuration
try:
    with open("config.yaml", "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f) or {}
except Exception as e:
    print(f"Warning: Config error: {e}, using defaults")
    CONFIG = {}

PREP = {**DEFAULT_PREP, **CONFIG.get("preprocessing", {})}

CLASSES = CONFIG.get("classes", ["0_Normal", "1_Pneumonia"])
IMG_SIZE = CONFIG.get("image_size", 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Enhanced_Preprocessing:
    """Advanced preprocessing pipeline for chest X-ray images"""
    
    def remove_text_annotations(self, img):
        """Remove text annotations from border regions"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        _, thresh = cv2.threshold(
            gray, PREP["text_threshold"], 255, cv2.THRESH_BINARY
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
                x < w * PREP["border_margin_x"]
                or x > w * (1 - PREP["border_margin_x"])
                or y < h * PREP["border_margin_y_top"]
                or y > h * PREP["border_margin_y_bottom"]
            )

            if is_border and area < h * w * PREP["max_text_area_ratio"]:
                cv2.rectangle(mask, (x, y), (x + cw, y + ch), 255, -1)

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
            m = PREP["lung_crop_margin"]
            return img[int(h * m): int(h * (1 - m)),
                       int(w * m): int(w * (1 - m))]

        pts = np.vstack(valid)
        x, y, cw, ch = cv2.boundingRect(pts)

        pad = 20
        return img[
            max(0, y - pad): min(h, y + ch + pad),
            max(0, x - pad): min(w, x + cw + pad)
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
            clipLimit=PREP["clahe_clip_limit"],
            tileGridSize=(PREP["clahe_tile_size"],) * 2
        )
        img = clahe.apply(gray)

        # Gamma correction
        img = np.power(img / 255.0, PREP["gamma_correction"]) * 255
        
        # Denoise
        img = cv2.fastNlMeansDenoising(
            img.astype(np.uint8),
            h=PREP["denoise_strength"]
        )

        # Sharpen
        blur = cv2.GaussianBlur(img, (0, 0), PREP["gaussian_sigma"])
        img = cv2.addWeighted(
            img,
            PREP["sharpen_alpha"],
            blur,
            PREP["sharpen_beta"],
            0
        )

        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))


def load_model(arch):
    """Load trained model and identify target layer for Grad-CAM"""
    print(f"\nLoading {arch} for Grad-CAM")

    if arch == "txv_densenet":
        if not XRV_AVAILABLE:
            sys.exit("Error: TorchXRayVision required for DenseNet model")

        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        num_ftrs = model.classifier.in_features
        
        # Match training architecture
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, len(CLASSES))
        )
        
        if hasattr(model, "op_threshs"):
            model.op_threshs = None
            
        weight_path = CONFIG.get(
            "model_txv_densenet",
            "outputs/models/txv_densenet_best.pth"
        )
        target_layers = [model.features[-1]]

    else:  # resnet50
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        
        # Match training architecture
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, len(CLASSES))
        )
        
        weight_path = CONFIG.get(
            "model_resnet",
            "outputs/models/resnet50_best.pth"
        )
        target_layers = [model.layer4[-1]]

    if not os.path.exists(weight_path):
        sys.exit(f"Error: Model weight not found: {weight_path}")

    try:
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        model.to(DEVICE).eval()
        print(f"   Loaded weights: {weight_path}")
    except Exception as e:
        sys.exit(f"Error: Failed to load model: {e}")

    return model, target_layers


def run_gradcam(args):
    """Generate and save Grad-CAM visualization"""
    model, target_layers = load_model(args.arch)

    # Define appropriate transform for architecture
    if args.arch == "txv_densenet":
        transform = transforms.Compose([
            Enhanced_Preprocessing(),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    else:
        transform = transforms.Compose([
            Enhanced_Preprocessing(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    # Load and preprocess image
    try:
        img_raw = Image.open(args.image).convert("RGB")
    except Exception as e:
        sys.exit(f"Error: Cannot open image: {e}")

    input_tensor = transform(img_raw).unsqueeze(0).to(DEVICE)

    # Get prediction
    with torch.no_grad():
        out = model(input_tensor)
        prob = torch.softmax(out, 1)
        cls_idx = prob.argmax(1).item()
        label = CLASSES[cls_idx]

    print(f"\nPrediction: {label} ({prob[0, cls_idx]*100:.2f}%)")

    # Generate Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    cam_map = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(cls_idx)]
    )[0]

    # Apply threshold to remove weak activations
    cam_map[cam_map < 0.2] = 0

    # Prepare display image
    display_img = np.array(Enhanced_Preprocessing()(img_raw))
    display_img = display_img.astype(np.float32) / 255.0

    # Overlay heatmap on image
    cam_img = show_cam_on_image(
        display_img,
        cam_map,
        use_rgb=True,
        image_weight=0.6
    )

    # Create side-by-side comparison
    final = np.hstack([
        (display_img * 255).astype(np.uint8),
        cam_img
    ])

    # Save result
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(
        args.output_dir,
        f"CAM_{args.arch}_{label}_{os.path.basename(args.image)}"
    )
    
    try:
        cv2.imwrite(save_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
        print(f"Saved: {save_path}\n")
    except Exception as e:
        print(f"Error: Failed to save: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--arch",
        default="txv_densenet",
        choices=["txv_densenet", "resnet50"],
        help="Model architecture"
    )
    parser.add_argument("--output_dir", default="outputs/gradcam", help="Output directory")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"Error: Image not found: {args.image}")

    try:
        run_gradcam(args)
    except Exception as e:
        print(f"\nError: Grad-CAM failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)