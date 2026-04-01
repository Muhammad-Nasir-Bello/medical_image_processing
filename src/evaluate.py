import os
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score, roc_auc_score
)

from src.config import CFG
from src.model import build_model
from src.data import make_loaders


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def load_best_model(ckpt_path):
    """Load best saved model weights."""
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location=CFG.device)
    model.load_state_dict(ckpt["model"])
    model.to(CFG.device)
    model.eval()
    return model


def evaluate():
    print("Loading test set...")
    _, _, test_loader = make_loaders()

    best_path = Path(CFG.out_dir) / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"No best.pt found at {best_path}")

    print(f"📦 Loading best model: {best_path}")
    model = load_best_model(best_path)

    all_probs, all_true = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(CFG.device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            all_probs.append(probs)
            all_true.append(labels.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    y_pred = y_prob.argmax(axis=1)

    # ---- Metrics ----
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    try:
        macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except:
        macro_auc = float("nan")

    out_dir = Path(CFG.out_dir)
    out_dir.mkdir(exist_ok=True)

    # ---- Save classification report ----
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Macro AUC: {macro_auc:.4f}\n\n")
        f.write(report)

    print("Saved: classification_report.txt")

    # ---- Confusion Matrix Plot ----
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)

    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix"
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")

    # ---- ROC Curve Plot ----
    plt.figure(figsize=(7, 6))
    for c in range(len(CLASS_NAMES)):
        y_true_c = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_true_c, y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{CLASS_NAMES[c]} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_ovr.png", dpi=150)
    plt.close()
    print("Saved: roc_ovr.png")

    # ---- Save Summary JSON ----
    summary = {
        "accuracy": acc,
        "macro_auc_ovr": macro_auc,
        "classes": CLASS_NAMES,
        "confusion_matrix": cm.tolist(),
    }

    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved: metrics_summary.json")

    print("\n Evaluation complete!")


if __name__ == "__main__":
    evaluate()
