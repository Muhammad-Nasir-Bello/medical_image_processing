# app.py
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image

from src.config import CFG
from src.model import build_model
from src.data import get_transforms
from src.gradcam import GradCAM, overlay_cam


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
)


# -----------------------------
# Cached model loader
# -----------------------------
@st.cache_resource
def load_model_and_cam():
    best_path = Path("experiments/run1/best.pt")
    if not best_path.exists():
        raise FileNotFoundError(
            "best.pt not found at experiments/run1/best.pt. "
            "Train first with: python -m src.train"
        )

    model = build_model()
    ckpt = torch.load(best_path, map_location=CFG.device)
    model.load_state_dict(ckpt["model"])
    model.to(CFG.device)
    model.eval()

    # Grad-CAM engine on final conv block (ResNet18.layer4[-1])
    target_layer = model.layer4[-1]
    cam_engine = GradCAM(model, target_layer)

    return model, cam_engine


@st.cache_resource
def load_metrics():
    summary_path = Path("experiments/run1/metrics_summary.json")
    if summary_path.exists():
        with open(summary_path, "r") as f:
            return json.load(f)
    return None


# -----------------------------
# Preprocessing & prediction
# -----------------------------
def preprocess_image(pil_img):
    """
    Takes a PIL image, applies the same transforms as training (val/test),
    and returns a (1, C, H, W) tensor on CFG.device.
    """
    img_rgb = np.array(pil_img.convert("RGB"))  # HWC, RGB
    tfms = get_transforms(train=False)
    tensor = tfms(image=img_rgb)["image"]       # (C, H, W)
    tensor = tensor.unsqueeze(0).to(CFG.device)
    return img_rgb, tensor


def predict(model, tensor):
    """
    Runs a forward pass and returns:
    - predicted index
    - probability vector (numpy, shape [num_classes])
    """
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def generate_gradcam(cam_engine, tensor, class_idx):
    """
    Generates Grad-CAM for a given input tensor and class index.
    Returns the CAM map as a numpy array [H, W] in [0, 1].
    """
    cam, _ = cam_engine.generate(tensor, class_idx=class_idx)
    return cam


# -----------------------------
# Sidebar: model & experiment info
# -----------------------------
st.sidebar.title("🧠 Brain Tumor Classifier")

st.sidebar.markdown(
    """
This demo uses a **ResNet-18** model fine-tuned on
MRI brain images to classify:

- Glioma  
- Meningioma  
- No Tumor  
- Pituitary Tumor
"""
)

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Model Info")
st.sidebar.write(f"Device: **{CFG.device}**")
st.sidebar.write(f"Input size: **{CFG.img_size}×{CFG.img_size}**")
st.sidebar.write(f"Classes: {', '.join(CFG.class_names)}")

metrics = load_metrics()
if metrics is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Test Performance")
    st.sidebar.write(f"Accuracy: **{metrics.get('accuracy', 0):.4f}**")
    st.sidebar.write(f"Macro AUC (OvR): **{metrics.get('macro_auc_ovr', 0):.4f}**")
else:
    st.sidebar.info("Run evaluation to populate metrics_summary.json")


st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ This tool is for **research and educational purposes only** and "
    "must **not** be used for real medical diagnosis."
)


# -----------------------------
# Main layout
# -----------------------------
st.title("🧠 MRI Brain Tumor Classification")
st.markdown(
    """
Upload a brain MRI image to see the **predicted tumor type**,  
**class probabilities**, and a **Grad-CAM heatmap** showing  
where the model is focusing.
"""
)

col1, col2 = st.columns([1, 1])

# File uploader
uploaded = st.file_uploader(
    "📤 Upload an MRI image (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    # Load model + GradCAM
    try:
        model, cam_engine = load_model_and_cam()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # Display original image
    pil_img = Image.open(uploaded)
    col1.subheader("Input MRI")
    col1.image(pil_img, use_container_width=True)

    # Preprocess & predict
    img_rgb, tensor = preprocess_image(pil_img)
    pred_idx, probs = predict(model, tensor)
    pred_label = CFG.class_names[pred_idx]
    confidence = probs[pred_idx]

    # Prediction block
    col2.subheader("Prediction")
    col2.markdown(
        f"### 🧩 Predicted class: **{pred_label.upper()}**  \n"
        f"Confidence: **{confidence:.4f}**"
    )

    prob_dict = {CFG.class_names[i]: float(probs[i]) for i in range(len(CFG.class_names))}
    col2.markdown("**Class probabilities:**")
    col2.bar_chart(prob_dict)

    # Grad-CAM
    st.markdown("---")
    st.subheader("🔍 Model Attention (Grad-CAM)")

    with st.spinner("Generating Grad-CAM heatmap..."):
        cam = generate_gradcam(cam_engine, tensor, class_idx=pred_idx)
        overlay = overlay_cam(img_rgb, cam)

    c1, c2 = st.columns([1, 1])
    c1.markdown("**Original MRI**")
    c1.image(pil_img, use_container_width=True)

    c2.markdown(f"**Grad-CAM for: {pred_label.upper()}**")
    c2.image(overlay, use_container_width=True)

else:
    st.info("👆 Upload an MRI image to begin.")