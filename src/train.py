import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

# Ensure src directory is in sys.path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

try:
    from src.model import UNet
    from src.dataset import get_dataloaders
    from src.metrics import CombinedBCEDiceLoss, calculate_metrics, THRESHOLD
except ModuleNotFoundError:
    from model import UNet
    from dataset import get_dataloaders
    from metrics import CombinedBCEDiceLoss, calculate_metrics, THRESHOLD


def train_model(
    dataset_root="data/Amazon Forest Dataset",
    output_dir="outputs",
    batch_size=2,
    epochs=12,
    learning_rate=1e-3,
    init_features=32,
    threshold=THRESHOLD
):
    print("=" * 60)
    print("AI-BASED FOREST COVER SEGMENTATION - U-NET TRAINING (CPU)")
    print("=" * 60)

    device = torch.device("cpu")
    print(f"Target Device        : {device}")
    print(f"Batch Size           : {batch_size}")
    print(f"Epochs               : {epochs}")
    print(f"Learning Rate        : {learning_rate}")
    print(f"Prediction Threshold : {threshold}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_loader, val_loader, _ = get_dataloaders(dataset_root=dataset_root, batch_size=batch_size)
    print(f"Train Dataset        : {len(train_loader.dataset)} images ({len(train_loader)} batches)")
    print(f"Validation Dataset   : {len(val_loader.dataset)} images ({len(val_loader)} batches)")

    # 2. Model, Loss, Optimizer
    model = UNet(in_channels=3, out_channels=1, init_features=init_features).to(device)
    criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_iou = 0.0
    history = []

    start_time = time.time()

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_metrics_sum = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            batch_metrics = calculate_metrics(logits, masks, threshold=threshold, is_logits=True)
            for key in train_metrics_sum:
                train_metrics_sum[key] += batch_metrics[key] * images.size(0)

        num_train_samples = len(train_loader.dataset)
        avg_train_loss = train_loss / num_train_samples
        avg_train_iou = train_metrics_sum["iou"] / num_train_samples
        avg_train_dice = train_metrics_sum["dice"] / num_train_samples

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_metrics_sum = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)

                val_loss += loss.item() * images.size(0)

                batch_metrics = calculate_metrics(logits, masks, threshold=threshold, is_logits=True)
                for key in val_metrics_sum:
                    val_metrics_sum[key] += batch_metrics[key] * images.size(0)

        num_val_samples = len(val_loader.dataset)
        avg_val_loss = val_loss / num_val_samples
        avg_val_iou = val_metrics_sum["iou"] / num_val_samples
        avg_val_dice = val_metrics_sum["dice"] / num_val_samples
        avg_val_prec = val_metrics_sum["precision"] / num_val_samples
        avg_val_rec = val_metrics_sum["recall"] / num_val_samples
        avg_val_acc = val_metrics_sum["accuracy"] / num_val_samples

        scheduler.step(avg_val_iou)
        epoch_elapsed = time.time() - epoch_start

        # Record History
        epoch_data = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_iou": avg_train_iou,
            "train_dice": avg_train_dice,
            "val_loss": avg_val_loss,
            "val_iou": avg_val_iou,
            "val_dice": avg_val_dice,
            "val_precision": avg_val_prec,
            "val_recall": avg_val_rec,
            "val_accuracy": avg_val_acc,
            "time_sec": epoch_elapsed
        }
        history.append(epoch_data)

        # Save Best Checkpoint
        is_best = avg_val_iou > best_val_iou
        if is_best:
            best_val_iou = avg_val_iou
            best_model_path = output_path / "best_unet.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": avg_val_iou,
                "val_dice": avg_val_dice,
                "threshold": threshold
            }, best_model_path)
            checkpoint_msg = " [BEST MODEL SAVED]"
        else:
            checkpoint_msg = ""

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_elapsed:.1f}s) | "
            f"Train Loss: {avg_train_loss:.4f}, IoU: {avg_train_iou:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f}, Dice: {avg_val_dice:.4f}"
            f"{checkpoint_msg}"
        )

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training completed in {total_time / 60:.2f} minutes.")
    print(f"Best Validation IoU: {best_val_iou:.4f}")

    # 4. Save History CSV
    history_df = pd.DataFrame(history)
    csv_path = output_path / "training_history.csv"
    history_df.to_csv(csv_path, index=False)
    print(f"Saved training history to: {csv_path}")

    # 5. Plot Training Curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color="blue", linewidth=2)
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="red", linewidth=2)
    plt.title("Loss Curves (BCE + Dice Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["train_iou"], label="Train IoU", color="blue", linewidth=2)
    plt.plot(history_df["epoch"], history_df["val_iou"], label="Val IoU", color="green", linewidth=2)
    plt.plot(history_df["epoch"], history_df["val_dice"], label="Val Dice (F1)", color="orange", linewidth=2, linestyle="--")
    plt.title("Segmentation Performance Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    curves_path = output_path / "training_curves.png"
    plt.savefig(curves_path, dpi=150)
    plt.close()

    print(f"Saved training curves to: {curves_path}")
    print("=" * 60)

    return model, history_df


if __name__ == "__main__":
    train_model(epochs=12, batch_size=2)
