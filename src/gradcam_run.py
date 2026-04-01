# src/gradcam_run.py
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

from src.config import CFG
from src.model import build_model
from src.gradcam import GradCAM, overlay_cam
from src.data import get_transforms


# ------------------------------
# Load model checkpoint
# ------------------------------
def load_best():
    best_path = Path("experiments/run1/best.pt")
    if not best_path.exists():
        raise FileNotFoundError("❌ best.pt not found in experiments/run1/")

    model = build_model()
    ckpt = torch.load(best_path, map_location=CFG.device)
    model.load_state_dict(ckpt["model"])
    model.to(CFG.device)
    model.eval()
    return model


# ------------------------------
# Main Script
# ------------------------------
def main():

    # If an image is passed from terminal → use it
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Else → randomly sample from processed test set
        test_dir = Path("data/processed/test")
        test_images = list(test_dir.rglob("*.jpg"))
        img_path = str(np.random.choice(test_images))

    print(f" Using image: {img_path}")

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(" Cannot read image. Check path.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocessing
    tfms = get_transforms(train=False)
    tensor = tfms(image=img_rgb)["image"].unsqueeze(0).to(CFG.device)

    # Load trained model
    model = load_best()

    # Prepare Grad-CAM on ResNet18 layer4
    target_layer = model.layer4[-1]
    cam_gen = GradCAM(model, target_layer)

    # Generate CAM
    cam, pred_idx = cam_gen.generate(tensor)
    class_name = CFG.class_names[pred_idx]

    # Overlay CAM (handles resizing inside overlay_cam)
    overlay = overlay_cam(img_rgb, cam)

    save_path = "experiments/run1/gradcam_sample.png"
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f" Grad-CAM saved → {save_path}")
    print(f" Predicted Class: {class_name}")


if __name__ == "__main__":
    main()
