# src/train.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from dataset import KvasirSegDataset
from model_unet import UNet

# -----------------------
# Config
# -----------------------
DATA_DIR = r"data\Kvasir-SEG"
OUT_DIR = r"outputs"
EXP_NAME = "unet_img320_lr3e-4_dice70"
OUT_DIR = os.path.join("outputs", EXP_NAME)
os.makedirs(OUT_DIR, exist_ok=True)


SEED = 42
IMG_SIZE = 320
BATCH_SIZE = 8
EPOCHS = 15
LR = 3e-4

# AMP (mixed precision) speeds up GPU training; safe to keep ON
USE_AMP = True

# -----------------------
# Utilities
# -----------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def soft_dice_score(logits, targets, eps=1e-6):
    """
    Soft dice computed on probabilities (no threshold).
    Returns a float (mean across batch).
    """
    probs = torch.sigmoid(logits)
    targets = targets.float()
    inter = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()

def iou_score(logits, targets, eps=1e-6):
    """
    IoU computed on thresholded predictions (0.5).
    Returns a float (mean across batch).
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = (targets > 0.5).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()

def soft_dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()
    inter = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()

def denorm_to_01(img_chw: np.ndarray) -> np.ndarray:
    """
    Albumentations Normalize() makes images approx N(0,1).
    For visualization we just scale each image to [0,1] per-image.
    """
    x = img_chw
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return x

def save_pred_samples(model, loader, device, out_path, max_images=3):
    model.eval()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    n = min(max_images, x.size(0))

    # x: (N,C,H,W) -> (N,H,W,C)
    x_np = x[:n].detach().cpu().numpy()
    x_np = np.transpose(x_np, (0, 2, 3, 1))
    y_np = y[:n, 0].detach().cpu().numpy()
    p_np = preds[:n, 0].detach().cpu().numpy()

    plt.figure(figsize=(12, 4 * n))
    for i in range(n):
        plt.subplot(n, 3, 3 * i + 1)
        plt.title("Image")
        plt.imshow(denorm_to_01(np.transpose(x[i].detach().cpu().numpy(), (1, 2, 0))))
        plt.axis("off")

        plt.subplot(n, 3, 3 * i + 2)
        plt.title("GT Mask")
        plt.imshow(y_np[i], cmap="gray")
        plt.axis("off")

        plt.subplot(n, 3, 3 * i + 3)
        plt.title("Pred Mask")
        plt.imshow(p_np[i], cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# -----------------------
# Training
# -----------------------
def main():
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Collect filenames
    img_dir = os.path.join(DATA_DIR, "images")
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=SEED)

    # Transforms
    train_tfms = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ColorJitter(p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ])

    val_tfms = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ])

    # Datasets / Loaders (num_workers=0 is safest on Windows)
    train_ds = KvasirSegDataset(DATA_DIR, train_files, transform=train_tfms)
    val_ds = KvasirSegDataset(DATA_DIR, val_files, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    # Model / loss / optimizer
    model = UNet().to(device)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # AMP scaler
    use_amp = (device == "cuda") and USE_AMP
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_dice = -1.0
    best_path = os.path.join(OUT_DIR, "best.pt")

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = 0.3 * bce(logits, y) + 0.7 * soft_dice_loss(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(1, len(train_loader))

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]")
            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                    loss = 0.5 * bce(logits, y) + 0.5 * soft_dice_loss(logits, y)

                val_loss += loss.item()
                val_dice += soft_dice_score(logits, y)
                val_iou += iou_score(logits, y)

        val_loss /= max(1, len(val_loader))
        val_dice /= max(1, len(val_loader))
        val_iou /= max(1, len(val_loader))

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_dice={val_dice:.4f}  "
            f"val_iou={val_iou:.4f}"
        )

        # Save sample predictions every epoch
        sample_path = os.path.join(OUT_DIR, f"epoch_{epoch:02d}_samples.png")
        save_pred_samples(model, val_loader, device, sample_path)

        # Save best checkpoint (by val_dice)
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "img_size": IMG_SIZE,
                    "epoch": epoch,
                    "best_val_dice": best_dice,
                },
                best_path,
            )
            print(f"Saved best checkpoint: {best_path} (val_dice={best_dice:.4f})")

    print("Training done.")
    print("Best val dice:", best_dice)

if __name__ == "__main__":
    main()
