# src/split_full_dataset.py

import os
import shutil
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

def gather_all_images():
    data = []

    for split in ["Training", "Testing"]:
        for cls in CLASSES:
            folder = os.path.join(RAW_DIR, split, cls)
            for f in os.listdir(folder):
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    data.append((os.path.join(folder, f), cls))

    return data


def prepare_dirs():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)


def copy_images(pairs, split):
    for src, label in pairs:
        fname = os.path.basename(src)
        dst = os.path.join(OUT_DIR, split, label, fname)
        shutil.copy(src, dst)


def main():
    print("Collecting all images...")
    images = gather_all_images()
    paths = [p[0] for p in images]
    labels = [CLASSES.index(p[1]) for p in images]

    print(f"Total images found: {len(paths)}")

    # 70% train, then remaining 30% → 50/50 into val/test = 15/15
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=42
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
    )

    print("Preparing directories...")
    prepare_dirs()

    # Convert back to paired form
    train_pairs = list(zip(train_paths, [CLASSES[l] for l in train_labels]))
    val_pairs = list(zip(val_paths, [CLASSES[l] for l in val_labels]))
    test_pairs = list(zip(test_paths, [CLASSES[l] for l in test_labels]))

    print("Copying images...")
    copy_images(train_pairs, "train")
    copy_images(val_pairs, "val")
    copy_images(test_pairs, "test")

    print("\n DONE! Full dataset split created.\n")
    print(f"Train: {len(train_paths)}")
    print(f"Val:   {len(val_paths)}")
    print(f"Test:  {len(test_paths)}")


if __name__ == "__main__":
    main()
