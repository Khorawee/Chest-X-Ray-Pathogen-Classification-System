import argparse
import os
import time
import random
import numpy as np
import yaml
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

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
    print(f"Warning: Config error: {e}, using defaults")
    CONFIG = {
        "seed": 42,
        "image_size": 224,
        "batch_size": 8,
        "num_workers": 0,
        "lr": 1e-4,
        "epochs": 15,
        "data_dir": "./data_masked",
        "model_txv_densenet": "outputs/models/txv_densenet_best.pth",
        "model_resnet": "outputs/models/resnet50_best.pth",
        "classes": ["0_Normal", "1_Pneumonia"],
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name):
    """Setup logger to write to both file and console"""
    os.makedirs("outputs/logs", exist_ok=True)
    log_file = f"outputs/logs/train_{name}_{time.strftime('%Y%m%d-%H%M%S')}.txt"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


def get_transforms(model_name):
    """Get data augmentation transforms for training and validation"""
    img_size = CONFIG.get("image_size", 224)

    if model_name == "txv_densenet":
        # Grayscale transforms for TorchXRayVision models
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(7),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        val_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    else:
        # RGB transforms for standard ImageNet models
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(7),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        val_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    return {"train": train_tf, "val": val_tf}


def train_model(model_name):
    """Main training function for a specific model architecture"""
    set_seed(CONFIG.get("seed", 42))
    logger = setup_logger(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info(f"TRAINING MODEL: {model_name.upper()}")
    logger.info(f"DEVICE: {device}")
    logger.info(f"SEED: {CONFIG.get('seed', 42)}")
    logger.info("=" * 60)

    # Verify data directories exist
    data_dir = CONFIG.get("data_dir", "./data_masked")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not os.path.exists(os.path.join(data_dir, "train")):
        raise FileNotFoundError(f"Train directory not found: {os.path.join(data_dir, 'train')}")
    
    if not os.path.exists(os.path.join(data_dir, "val")):
        raise FileNotFoundError(f"Val directory not found: {os.path.join(data_dir, 'val')}")

    # Load datasets
    transforms_dict = get_transforms(model_name)

    try:
        datasets_dict = {
            x: datasets.ImageFolder(os.path.join(data_dir, x), transforms_dict[x])
            for x in ["train", "val"]
        }
    except Exception as e:
        raise RuntimeError(f"Failed to load datasets: {e}")

    dataloaders = {
        x: DataLoader(
            datasets_dict[x],
            batch_size=CONFIG.get("batch_size", 8),
            shuffle=(x == "train"),
            num_workers=CONFIG.get("num_workers", 0),
            pin_memory=(device.type == "cuda"),
        )
        for x in ["train", "val"]
    }

    dataset_sizes = {x: len(datasets_dict[x]) for x in ["train", "val"]}
    class_names = datasets_dict["train"].classes
    
    logger.info(f"Classes: {class_names}")
    logger.info(f"Train samples: {dataset_sizes['train']}")
    logger.info(f"Val samples: {dataset_sizes['val']}")
    logger.info(f"Batch size: {CONFIG.get('batch_size', 8)}")
    logger.info(f"Learning rate: {CONFIG.get('lr', 1e-4)}")
    logger.info(f"Epochs: {CONFIG.get('epochs', 15)}")

    # Initialize model
    if model_name == "txv_densenet":
        if not XRV_AVAILABLE:
            raise RuntimeError("torchxrayvision not installed")

        logger.info("\nLoading TorchXRayVision DenseNet...")
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, len(class_names))
        )
        
        if hasattr(model, "op_threshs"):
            model.op_threshs = None
            
        save_path = CONFIG.get("model_txv_densenet", "outputs/models/txv_densenet_best.pth")
        logger.info(f"   Classifier: Dropout(0.5) + Linear({num_ftrs} -> {len(class_names)})")

    else:
        logger.info("\nLoading ResNet50...")
        model = models.resnet50(weights="IMAGENET1K_V2")
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, len(class_names))
        )
        save_path = CONFIG.get("model_resnet", "outputs/models/resnet50_best.pth")
        logger.info(f"   Classifier: Dropout(0.5) + Linear({num_ftrs} -> {len(class_names)})")

    model.to(device)
    logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer and scheduler
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG.get("lr", 1e-4), 
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG.get("epochs", 15)
    )

    best_acc = 0.0
    best_epoch = 0
    os.makedirs("outputs/models", exist_ok=True)

    # Training loop
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    
    for epoch in range(CONFIG.get("epochs", 15)):
        logger.info(f"\nEpoch {epoch+1}/{CONFIG.get('epochs', 15)}")
        logger.info("-" * 60)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            pbar = tqdm(
                dataloaders[phase], 
                desc=f"{phase.upper():5s}", 
                ncols=100,
                leave=False
            )

            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{(preds == labels).float().mean():.4f}'
                })

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.info(
                f"[{phase.upper():5s}] Loss: {epoch_loss:.4f} | "
                f"Acc: {epoch_acc:.4f} ({running_corrects}/{dataset_sizes[phase]})"
            )

            # Save best model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch + 1
                try:
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"Saved best model: {save_path} (Acc={best_acc:.4f})")
                except Exception as e:
                    logger.error(f"Failed to save model: {e}")

    logger.info("\n" + "=" * 60)
    logger.info(f"TRAINING COMPLETED: {model_name.upper()}")
    logger.info(f"   Best Accuracy: {best_acc:.4f} (Epoch {best_epoch})")
    logger.info(f"   Model saved: {save_path}")
    logger.info("=" * 60 + "\n")


def main():
    """Main entry point for training script"""
    parser = argparse.ArgumentParser(description="Train chest X-ray classification models")
    parser.add_argument(
        "--model", 
        default="all",
        choices=["txv_densenet", "resnet50", "all"],
        help="Model to train"
    )
    args = parser.parse_args()

    try:
        if args.model == "all":
            print("\nTraining both models sequentially...\n")
            train_model("txv_densenet")
            print("\n" + "="*60 + "\n")
            train_model("resnet50")
        else:
            train_model(args.model)
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())