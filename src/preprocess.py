import os
import cv2
import numpy as np
import glob
import yaml
from tqdm import tqdm

# Load configuration
CONFIG_PATH = "config.yaml"

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError("config.yaml is empty")
except Exception as e:
    print(f"Error loading config: {e}")
    print("Using default values...")
    cfg = {
        "classes": ["0_Normal", "1_Pneumonia"],
        "image_size": 224,
        "preprocessing": {
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
    }

CLASSES = cfg.get("classes", ["0_Normal", "1_Pneumonia"])
IMG_SIZE = cfg.get("image_size", 224)
PREP = cfg.get("preprocessing", {})

INPUT_ROOT = "./data"
OUTPUT_ROOT = "./data_masked"
SUBSETS = ["train", "val", "test"]


def remove_text_and_border(img):
    """
    Remove text annotations and border artifacts
    Matches Enhanced_Preprocessing.remove_text_annotations()
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Threshold to detect bright regions (text)
    _, thresh = cv2.threshold(
        gray,
        PREP.get("text_threshold", 240),
        255,
        cv2.THRESH_BINARY
    )

    # Morphological operations to connect nearby text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    mask = np.zeros_like(gray)

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # Check if near border
        is_border = (
            x < w * PREP.get("border_margin_x", 0.08) or
            x > w * (1 - PREP.get("border_margin_x", 0.08)) or
            y < h * PREP.get("border_margin_y_top", 0.12) or
            y > h * PREP.get("border_margin_y_bottom", 0.88)
        )

        # Only remove small regions near borders (text annotations)
        if is_border and area < h * w * PREP.get("max_text_area_ratio", 0.02):
            cv2.rectangle(mask, (x, y), (x + cw, y + ch), 255, -1)

    # Inpaint the masked regions
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def segment_lung_region(img):
    """
    Advanced lung segmentation using morphological operations
    Matches Enhanced_Preprocessing.segment_lung_region()
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Apply Gaussian blur before thresholding
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's thresholding for automatic threshold selection
    _, bin_img = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological closing to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find largest contours (lung regions)
    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter valid lung regions (> 5% of image area)
    valid_contours = [c for c in contours if cv2.contourArea(c) > h * w * 0.05]

    if not valid_contours:
        # Fallback: simple crop with margin
        m = PREP.get("lung_crop_margin", 0.05)
        return img[int(h*m):int(h*(1-m)), int(w*m):int(w*(1-m))]

    # Get bounding box of all valid contours
    all_points = np.vstack(valid_contours)
    x, y, cw, ch = cv2.boundingRect(all_points)

    # Add padding
    pad = 20
    return img[
        max(0, y-pad):min(h, y+ch+pad),
        max(0, x-pad):min(w, x+cw+pad)
    ]


def enhance_image(img):
    """
    Apply medical image enhancement pipeline
    Matches Enhanced_Preprocessing enhancement steps
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(
        clipLimit=PREP.get("clahe_clip_limit", 0.6),
        tileGridSize=(PREP.get("clahe_tile_size", 8),) * 2
    )
    gray = clahe.apply(gray)

    # Gamma correction
    gamma = PREP.get("gamma_correction", 1.0)
    if gamma != 1.0:
        gray = np.power(gray / 255.0, gamma) * 255
        gray = gray.astype(np.uint8)

    # Denoising
    denoise_strength = PREP.get("denoise_strength", 2)
    if denoise_strength > 0:
        gray = cv2.fastNlMeansDenoising(
            gray,
            h=denoise_strength
        )

    # Gaussian blur for sharpening
    blur = cv2.GaussianBlur(
        gray,
        (0, 0),
        PREP.get("gaussian_sigma", 0.6)
    )

    # Unsharp masking (subtle sharpening)
    sharpened = cv2.addWeighted(
        gray,
        PREP.get("sharpen_alpha", 1.1),
        blur,
        PREP.get("sharpen_beta", -0.1),
        0
    )

    # Convert back to BGR for consistency
    return cv2.cvtColor(
        np.clip(sharpened, 0, 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )


def process_image(img):
    """
    Complete preprocessing pipeline
    Matches the full Enhanced_Preprocessing workflow
    """
    img = remove_text_and_border(img)
    img = segment_lung_region(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = enhance_image(img)
    return img


def run():
    """Batch process all images in dataset"""
    print("=" * 60)
    print("CXR PREPROCESSING (Enhanced Pipeline)")
    print("=" * 60)
    print(f"Input:  {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print("=" * 60)

    total = 0
    errors = 0

    for subset in SUBSETS:
        for cls in CLASSES:
            in_dir = os.path.join(INPUT_ROOT, subset, cls)
            out_dir = os.path.join(OUTPUT_ROOT, subset, cls)

            if not os.path.exists(in_dir):
                print(f"\nSkip: {subset}/{cls} (not found)")
                continue

            os.makedirs(out_dir, exist_ok=True)
            images = glob.glob(os.path.join(in_dir, "*.png")) + \
                     glob.glob(os.path.join(in_dir, "*.jpg")) + \
                     glob.glob(os.path.join(in_dir, "*.jpeg"))

            if not images:
                print(f"\nSkip: {subset}/{cls} (no images)")
                continue

            print(f"\n{subset}/{cls}: {len(images)} images")

            for img_path in tqdm(images, ncols=80, desc=f"{subset}/{cls}"):
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError("Cannot read image")

                    out = process_image(img)
                    
                    out_path = os.path.join(out_dir, os.path.basename(img_path))
                    cv2.imwrite(out_path, out)
                    total += 1
                    
                except Exception as e:
                    errors += 1
                    print(f"\nError [{os.path.basename(img_path)}]: {e}")

    print("\n" + "=" * 60)
    print(f"DONE!")
    print(f"   Processed: {total} images")
    print(f"   Errors:    {errors} images")
    print(f"   Output:    {OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    run()