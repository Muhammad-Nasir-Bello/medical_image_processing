import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import json

from src.data import make_loaders
from src.model import build_model
from src.config import CFG


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    correct = 0
    n = 0

    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += imgs.size(0)

    return total_loss / n, correct / n


def validate_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    n = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n += imgs.size(0)

    return total_loss / n, correct / n


def plot_curves(train_loss, val_loss, val_acc, out_dir):
    # Loss curve
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(out_dir / "loss_curve.png")
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(val_acc, label="Val Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.savefig(out_dir / "acc_curve.png")
    plt.close()


def main():
    device = CFG.device
    out_dir = Path(CFG.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("📦 Loading data...")
    train_loader, val_loader, _ = make_loaders()

    print("🧠 Building model...")
    model = build_model().to(device)

    optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0
    train_losses, val_losses, val_accs = [], [], []

    print("🚀 Starting training...")
    for epoch in range(CFG.epochs):
        print(f"\nEpoch {epoch+1}/{CFG.epochs}")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, loss_fn, device)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict()}, out_dir / "best.pt")
            print("💾 Saved new BEST model!")

    # Save curves
    plot_curves(train_losses, val_losses, val_accs, out_dir)

    # Save metrics summary
    summary = {
        "best_val_acc": best_acc,
        "epochs": CFG.epochs
    }
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()