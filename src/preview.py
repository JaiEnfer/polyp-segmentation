import os, random
from PIL import Image
import matplotlib.pyplot as plt


DATA_DIR = r".\data\Kvasir-SEG"
IMG_DIR = os.path.join(DATA_DIR, "images")
MSK_DIR = os.path.join(DATA_DIR, "masks")

imgs = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg",".jpeg", ".png"))]
print("images: ", len(imgs))


img_name = random.choice(imgs)
img = Image.open(os.path.join(IMG_DIR, img_name)).convert("RGB")
mask = Image.open(os.path.join(MSK_DIR, img_name)).convert("L")

plt.figure(figsize= (10,4))
plt.subplot(1,2,1); plt.title("Image"); plt.imshow(img); plt.axis("off")
plt.subplot(1,2,2); plt.title("Mask"); plt.imshow(mask, cmap="gray"); plt.axis("off")
plt.tight_layout()
plt.show()