# src/config.py
import torch
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CFG:
    # Paths
    dataset_dir: str = "data/processed"          # <-- we now use the processed splits
    out_dir: str = "experiments/run1"

    # Data
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    num_classes: int = 4
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]

    # Training
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-4

    # Device (M2/M3 → MPS, else CUDA/CPU)
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    seed: int = 42

    @staticmethod
    def ensure_dirs():
        Path("experiments").mkdir(exist_ok=True)
        Path("experiments/run1").mkdir(parents=True, exist_ok=True)

CFG.ensure_dirs()