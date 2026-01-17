# src/evaluate.py
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import KvasirSegDataset
from model_unet import UNet


def soft_dice_score_from_probs(probs, targets, eps=1e-6):
    # probs, targets: (B,1,H,W)
    inter = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


def iou_score_from_probs(probs, targets, thr=0.5, eps=1e-6):
    preds = (probs > thr).float()
    targets = (targets > 0.5).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()


def denorm_to_01(x_hwc: np.ndarray) -> np.ndarray:
    x = x_hwc
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def save_visuals(model, loader, device, out_dir, thr=0.5, num_images=12):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    saved = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = torch.sigmoid(model(x))
            preds = (probs > thr).float()

            b = x.size(0)
            for i in range(b):
                if saved >= num_images:
                    return

                # x is normalized; for display, scale each image to 0-1
                img = x[i].detach().cpu().numpy().transpose(1, 2, 0)
                img = denorm_to_01(img)
                gt = y[i, 0].detach().cpu().numpy()
                pr = preds[i, 0].detach().cpu().numpy()

                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1); plt.title("Image"); plt.imshow(img); plt.axis("off")
                plt.subplot(1, 3, 2); plt.title("GT"); plt.imshow(gt, cmap="gray"); plt.axis("off")
                plt.subplot(1, 3, 3); plt.title(f"Pred (thr={thr})"); plt.imshow(pr, cmap="gray"); plt.axis("off")
                plt.tight_layout()

                out_path = os.path.join(out_dir, f"val_pred_{saved:03d}.png")
                plt.savefig(out_path, dpi=150)
                plt.close()
                saved += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"data\Kvasir-SEG")
    ap.add_argument("--ckpt", default=r"outputs\unet_kvasir_best.pt")
    ap.add_argument("--img_size", type=int, default=None, help="Override; otherwise read from checkpoint if present.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--save_dir", default=r"outputs\eval_visuals", help="Folder to save example predictions.")
    ap.add_argument("--save_n", type=int, default=12, help="How many val images to save as visuals.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ckpt = torch.load(args.ckpt, map_location=device)
    img_size = args.img_size or ckpt.get("img_size", 256)

    model = UNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # deterministic val split (same as training split logic)
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

    soft_dice_total = 0.0
    iou_total = 0.0

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            probs = torch.sigmoid(model(x))
            soft_dice_total += soft_dice_score_from_probs(probs, y)
            iou_total += iou_score_from_probs(probs, y, thr=args.thr)

    soft_dice = soft_dice_total / len(val_loader)
    iou = iou_total / len(val_loader)

    print(f"Checkpoint: {args.ckpt}")
    print(f"IMG_SIZE: {img_size}")
    print(f"Val Soft-Dice (no threshold): {soft_dice:.4f}")
    print(f"Val IoU @ thr={args.thr:.2f}: {iou:.4f}")

    if args.save_n > 0:
        save_visuals(model, val_loader, device, args.save_dir, thr=args.thr, num_images=args.save_n)
        print(f"Saved {args.save_n} visuals to: {args.save_dir}")


if __name__ == "__main__":
    main()
