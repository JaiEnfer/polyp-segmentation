# src/threshold_sweep.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import KvasirSegDataset
from model_unet import UNet


def dice_at_threshold(probs, targets, thr=0.5, eps=1e-6):
    preds = (probs > thr).float()
    targets = (targets > 0.5).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


def iou_at_threshold(probs, targets, thr=0.5, eps=1e-6):
    preds = (probs > thr).float()
    targets = (targets > 0.5).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"data\Kvasir-SEG")
    ap.add_argument("--ckpt", default=r"outputs\unet_kvasir_best.pt")
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tmin", type=float, default=0.30)
    ap.add_argument("--tmax", type=float, default=0.70)
    ap.add_argument("--tstep", type=float, default=0.05)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ckpt = torch.load(args.ckpt, map_location=device)
    img_size = args.img_size or ckpt.get("img_size", 256)

    model = UNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img_dir = os.path.join(args.data_dir, "images")
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    _, val_files = train_test_split(files, test_size=0.2, random_state=args.seed)

    val_tfms = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2(),
    ])

    val_ds = KvasirSegDataset(args.data_dir, val_files, transform=val_tfms)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    thresholds = []
    t = args.tmin
    while t <= args.tmax + 1e-9:
        thresholds.append(round(t, 4))
        t += args.tstep

    best_dice = (-1.0, None)
    best_iou = (-1.0, None)

    with torch.no_grad():
        for thr in thresholds:
            dice_sum = 0.0
            iou_sum = 0.0

            for x, y in tqdm(val_loader, desc=f"thr={thr:.2f}", leave=False):
                x, y = x.to(device), y.to(device)
                probs = torch.sigmoid(model(x))
                dice_sum += dice_at_threshold(probs, y, thr=thr)
                iou_sum += iou_at_threshold(probs, y, thr=thr)

            mean_dice = dice_sum / len(val_loader)
            mean_iou = iou_sum / len(val_loader)

            print(f"thr={thr:.2f} | Dice={mean_dice:.4f} | IoU={mean_iou:.4f}")

            if mean_dice > best_dice[0]:
                best_dice = (mean_dice, thr)
            if mean_iou > best_iou[0]:
                best_iou = (mean_iou, thr)

    print("\nBest thresholds:")
    print(f"Best Dice: {best_dice[0]:.4f} at thr={best_dice[1]:.2f}")
    print(f"Best IoU : {best_iou[0]:.4f} at thr={best_iou[1]:.2f}")


if __name__ == "__main__":
    main()
