import io
import os
import numpy as np
import streamlit as st
from PIL import Image

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.model_unet import UNet  # make sure your model file is in src/model_unet.py

st.set_page_config(page_title="Polyp Segmentation UI (Kvasir-SEG)", layout="wide")

DEFAULT_CKPT = r"outputs\unet_img320_lr3e-4_dice70\best.pt"

def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    img_size = ckpt.get("img_size", 320)

    model = UNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, img_size

def preprocess(pil_img: Image.Image, img_size: int):
    tfm = A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()])
    img_np = np.array(pil_img.convert("RGB"))
    aug = tfm(image=img_np)
    x = aug["image"].unsqueeze(0)  # (1,3,H,W)
    return x, img_np

def predict_mask(model, x, device: str):
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()  # (H,W)
    return probs

def overlay_mask(rgb_img: np.ndarray, mask01: np.ndarray, alpha: float = 0.45):
    # rgb_img: H,W,3 uint8 ; mask01: H,W in {0,1}
    img = rgb_img.astype(np.float32)
    overlay = img.copy()
    # highlight mask area in red channel
    overlay[..., 0] = np.clip(overlay[..., 0] + 120 * mask01, 0, 255)
    out = (alpha * overlay + (1 - alpha) * img).astype(np.uint8)
    return out

def to_png_bytes(pil_img: Image.Image):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

st.title("🩺 Polyp Segmentation (U-Net) — Demo UI")
st.caption("Non-diagnostic Demo. Upload an endoscopy image and generate a segmentation mask.")

with st.sidebar:
    st.header("Model Settings")
    ckpt_path = st.text_input("Checkpoint path", value=DEFAULT_CKPT)
    threshold = st.slider("Threshold", min_value=0.05, max_value=0.95, value=0.30, step=0.01)
    overlay_alpha = st.slider("Overlay strength", 0.0, 0.9, 0.45, 0.05)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"**Device:** {device}")

    load_btn = st.button("Load / Reload model")

if "model" not in st.session_state or load_btn:
    if not os.path.exists(ckpt_path):
        st.error(f"Checkpoint not found: {ckpt_path}")
        st.stop()
    st.session_state.model, st.session_state.img_size = load_model(ckpt_path, device)
    st.success(f"Loaded model. Input size: {st.session_state.img_size}×{st.session_state.img_size}")

uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a polyp image to begin.")
    st.stop()

pil = Image.open(uploaded).convert("RGB")
x, rgb_np = preprocess(pil, st.session_state.img_size)
probs = predict_mask(st.session_state.model, x, device)

pred01 = (probs > threshold).astype(np.uint8)
pred_mask_pil = Image.fromarray((pred01 * 255).astype(np.uint8))
overlay = overlay_mask(np.array(pil.convert("RGB").resize((st.session_state.img_size, st.session_state.img_size))), pred01, alpha=overlay_alpha)
overlay_pil = Image.fromarray(overlay)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Input")
    st.image(pil, use_container_width=True)

with col2:
    st.subheader(f"Predicted Mask (thr={threshold:.2f})")
    st.image(pred_mask_pil, use_container_width=True)

with col3:
    st.subheader("Overlay")
    st.image(overlay_pil, use_container_width=True)

st.divider()
st.subheader("Downloads")
st.download_button("Download mask (PNG)", data=to_png_bytes(pred_mask_pil), file_name="mask.png", mime="image/png")
st.download_button("Download overlay (PNG)", data=to_png_bytes(overlay_pil), file_name="overlay.png", mime="image/png")
