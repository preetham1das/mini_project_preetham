import random
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
    from src.model import CHECKPOINT_PATH, UNet
    from src.dataset import IMAGE_SIZE, SEED, get_dataloaders
    from src.metrics import CombinedBCEDiceLoss, calculate_confusion_counts, metrics_from_counts, THRESHOLD
except ModuleNotFoundError:
    from model import CHECKPOINT_PATH, UNet
    from dataset import IMAGE_SIZE, SEED, get_dataloaders
    from metrics import CombinedBCEDiceLoss, calculate_confusion_counts, metrics_from_counts, THRESHOLD


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    dataset_root="data/Amazon Forest Dataset",
    output_dir="outputs",
    batch_size=2,
    epochs=12,
    learning_rate=1e-3,
    init_features=32,
    threshold=THRESHOLD,
    image_size=IMAGE_SIZE,
    seed=SEED,
    checkpoint_path=CHECKPOINT_PATH,
    early_stopping_patience=5
):
    print("=" * 60)
    print("AI-BASED FOREST COVER SEGMENTATION - U-NET TRAINING (CPU)")
    print("=" * 60)

    set_seed(seed)
    device = torch.device("cpu")
    print(f"Target Device        : {device}")
    print(f"Batch Size           : {batch_size}")
    print(f"Epochs               : {epochs}")
    print(f"Learning Rate        : {learning_rate}")
    print(f"Prediction Threshold : {threshold}")
    print(f"Training Image Size  : {image_size}x{image_size}")
    print(f"Random Seed          : {seed}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_loader, val_loader, _ = get_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        image_size=image_size
    )
    print(f"Train Dataset        : {len(train_loader.dataset)} images ({len(train_loader)} batches)")
    print(f"Validation Dataset   : {len(val_loader.dataset)} images ({len(val_loader)} batches)")

    # 2. Model, Loss, Optimizer
    model = UNet(in_channels=3, out_channels=1, init_features=init_features).to(device)
    criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_iou = -1.0
    epochs_without_improvement = 0
    history = []
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            batch_counts = calculate_confusion_counts(logits, masks, threshold=threshold, is_logits=True)
            for key in train_counts:
                train_counts[key] += batch_counts[key]

        num_train_samples = len(train_loader.dataset)
        avg_train_loss = train_loss / num_train_samples
        train_metrics = metrics_from_counts(train_counts)
        avg_train_iou = train_metrics["iou"]
        avg_train_dice = train_metrics["dice"]

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)

                val_loss += loss.item() * images.size(0)

                batch_counts = calculate_confusion_counts(logits, masks, threshold=threshold, is_logits=True)
                for key in val_counts:
                    val_counts[key] += batch_counts[key]

        num_val_samples = len(val_loader.dataset)
        avg_val_loss = val_loss / num_val_samples
        val_metrics = metrics_from_counts(val_counts)
        avg_val_iou = val_metrics["iou"]
        avg_val_dice = val_metrics["dice"]
        avg_val_prec = val_metrics["precision"]
        avg_val_rec = val_metrics["recall"]
        avg_val_acc = val_metrics["accuracy"]

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
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": avg_val_iou,
                "val_dice": avg_val_dice,
                "threshold": threshold,
                "image_size": image_size,
                "seed": seed
            }, checkpoint_file)
            checkpoint_msg = " [BEST MODEL SAVED]"
            epochs_without_improvement = 0
        else:
            checkpoint_msg = ""
            epochs_without_improvement += 1

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_elapsed:.1f}s) | "
            f"Train Loss: {avg_train_loss:.4f}, IoU: {avg_train_iou:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f}, Dice: {avg_val_dice:.4f}"
            f"{checkpoint_msg}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping after {epoch} epochs without validation IoU improvement.")
            break

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
