import os, random
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model_unet import UNet

DATA_DIR = r"data\Kvasir-SEG"
CKPT = CKPT = r"outputs\unet_img320_lr3e-4_dice70\best.pt"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT, map_location=device)
    img_size = ckpt.get("img_size", 256)

    model = UNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tfm = A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()])

    img_dir = os.path.join(DATA_DIR, "images")
    msk_dir = os.path.join(DATA_DIR, "masks")
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
    f = random.choice(files)

    img = np.array(Image.open(os.path.join(img_dir, f)).convert("RGB"))
    msk = np.array(Image.open(os.path.join(msk_dir, f)).convert("L"))
    msk = (msk > 0).astype(np.uint8)

    aug = tfm(image=img, mask=msk)
    x = aug["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0,0].cpu().numpy()
        THR = 0.30
        pred = (prob > THR).astype(np.uint8)


    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.title("Image"); plt.imshow(img); plt.axis("off")
    plt.subplot(1,3,2); plt.title("GT Mask"); plt.imshow(msk, cmap="gray"); plt.axis("off")
    plt.subplot(1,3,3); plt.title("Pred Mask"); plt.imshow(pred, cmap="gray"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(r"outputs\sample_prediction.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
