![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-enabled-76B900?style=flat&logo=nvidia&logoColor=white)

# 🩺 Polyp Segmentation with U-Net (Kvasir-SEG)

End-to-end **medical image segmentation (non-diagnostic)** project using a U-Net trained on the **Kvasir-SEG** dataset.  
The project covers the full ML lifecycle: training, evaluation, threshold optimization, inference, and an interactive web UI.

> ⚠️ **Disclaimer**  
> This project is for **research and demonstration purposes only**.  
> It is **not intended for clinical or diagnostic use**.

---

## ✨ Features
- U-Net implemented **from scratch** in PyTorch
- Dice + BCE loss for segmentation-focused optimization
- CUDA-accelerated training
- Validation-based model selection
- Threshold optimization for best Dice / IoU
- Quantitative + qualitative evaluation
- Interactive **Streamlit UI** for inference
- ONNX export for deployment readiness

---

## 📊 Dataset
**Kvasir-SEG** (polyp segmentation)

- 1,000 colonoscopy images
- Pixel-wise binary masks
- RGB images

🔗 **Download**:  
{https://datasets.simula.no/kvasir-seg/}

After extraction, place the data as:
```text
data/Kvasir-SEG/images
data/Kvasir-SEG/masks
```
📌 The dataset is **not included** in this repository due to size and licensing considerations.

---

## 🏗️ Model & Training Details
- **Architecture:** U-Net
- **Input size:** 320 × 320
- **Loss:** 0.7 Dice + 0.3 BCE
- **Optimizer:** Adam
- **Learning rate:** 3e-4
- **Batch size:** 4–8 (GPU dependent)
- **Hardware:** NVIDIA GPU with CUDA support

The **best model** is selected automatically based on **validation soft-Dice score**.

---

## 🏆 Results (Validation Set)

| Metric | Value |
|------|------|
| **Soft-Dice (threshold-free)** | **0.5604** |
| **Best Dice (thresholded)** | **0.5967 @ threshold = 0.30** |
| **Best IoU (thresholded)** | **0.4675 @ threshold = 0.45** |

📁 Example predictions:
```text
outputs/unet_img320_lr3e-4_dice70/eval_visuals/
```

---

## 🚀 Setup (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
---
## Training 
```sh
python src/train.py
```
Artifacts are saved under: 
```text
outputs/<experiment_name>/
```
---
## 📈 Evaluation

```sh
python src/evaluate.py --ckpt outputs/unet_img320_lr3e-4_dice70/best.pt
python src/threshold_sweep.py --ckpt outputs/unet_img320_lr3e-4_dice70/best.pt
```
## 🔍 Inference
```sh
python src/infer.py
```
## 🖥️ Web UI (Streamlit Demo)

An interactive UI for uploading images, adjusting thresholds, and visualizing predictions.

```sh
streamlit run app.py
```
### Features:
- Image upload
- Threshold slider
- Mask & overlay visualization
- Mask and overlay download

---
## 📦 ONNX Export
```sh
python src/export_onnx.py
```
Exports the trained U-Net model for deployment using ONNX-compatible runtimes.

---
## 🧠 Key ML Practices Demonstrated

1. Proper train/validation split with fixed seed
2. Metric-driven model selection
3. Threshold optimization on validation set
4. Visual inspection of segmentation quality
5. GPU-accelerated training
6. Reproducible experiments

---

##🛡️ Ethical & Legal Notes

- Non-diagnostic research project
- Dataset not redistributed
- No patient-identifiable data included
- GDPR-aware data handling

---
##⭐ Why this project matters

This project demonstrates:

1. Real-world medical computer vision
2. Strong segmentation fundamentals
3. Clean evaluation methodology
4. Deployment mindset with UI + ONNX


---
### UI Screenshot
<img width="1907" height="878" alt="image" src="https://github.com/user-attachments/assets/93ab1117-85a9-4ec7-891c-cc9bdc20312c" />

<img width="1912" height="868" alt="image" src="https://github.com/user-attachments/assets/ddf4078f-33fa-4d0d-9426-cb25bd0ff148" />

---

___Thank You___




