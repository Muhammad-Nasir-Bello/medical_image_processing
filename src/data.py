# src/data.py
import os
from typing import Tuple, List

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import CFG


def get_transforms(train: bool = True):
    """Albumentations transforms for train / val / test."""
    if train:
        return A.Compose(
            [
                A.LongestMaxSize(CFG.img_size),
                A.PadIfNeeded(CFG.img_size, CFG.img_size, border_mode=cv2.BORDER_CONSTANT),
                A.RandomRotate90(p=0.3),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.LongestMaxSize(CFG.img_size),
                A.PadIfNeeded(CFG.img_size, CFG.img_size, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(),
                ToTensorV2(),
            ]
        )


class MRIDataset(Dataset):
    """
    Folder structure expected:
        data/processed/train/glioma/*.jpg
        data/processed/train/meningioma/*.jpg
        ...
        data/processed/val/...
        data/processed/test/...
    """

    def __init__(self, root_dir: str, class_names: List[str], train: bool):
        self.root_dir = root_dir
        self.class_names = class_names
        self.tfms = get_transforms(train)

        self.samples: List[Tuple[str, int]] = []

        for label, cls_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Class folder not found: {class_dir}")

            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(class_dir, fname), label)
                    )

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = self.tfms(image=img)["image"]

        label = torch.tensor(label, dtype=torch.long)
        return img, label


def make_loaders():
    base = CFG.dataset_dir  # "data/processed"

    train_dir = os.path.join(base, "train")
    val_dir = os.path.join(base, "val")
    test_dir = os.path.join(base, "test")

    class_names = CFG.class_names

    train_ds = MRIDataset(train_dir, class_names, train=True)
    val_ds = MRIDataset(val_dir, class_names, train=False)
    test_ds = MRIDataset(test_dir, class_names, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_ds)} images")
    print(f"Val:   {len(val_ds)} images")
    print(f"Test:  {len(test_ds)} images")

    return train_loader, val_loader, test_loader