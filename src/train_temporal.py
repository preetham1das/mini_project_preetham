import sys
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Ensure src directory is in sys.path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from temporal_dataset import get_temporal_dataloaders
from temporal_model import TemporalUNet, TEMPORAL_CHECKPOINT_PATH
from metrics import CombinedBCEDiceLoss, calculate_metrics

# Hyperparameters
NUM_EPOCHS = 10  # Mini-project quick training
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
DEVICE = torch.device("cpu")

def main():
    print("=" * 60)
    print("STAGE 2: TEMPORAL DEFORESTATION DETECTION TRAINING")
    print("=" * 60)

    # 1. Dataloaders
    train_loader, val_loader, _ = get_temporal_dataloaders(batch_size=BATCH_SIZE)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Calculate class imbalance weighting for training data
    # Deforestation pixels are very rare compared to background.
    # We approximate from a few batches or assume a reasonable default.
    # For a mini-project, pos_weight ~ 10-20 is often appropriate for change detection.
    # We will compute it dynamically from the first few batches.
    print("Calculating positive class weight from sample of training data...")
    total_pos = 0
    total_neg = 0
    for i, (_, masks, _) in enumerate(train_loader):
        if i >= 5: break  # Sample 5 batches
        total_pos += masks.sum().item()
        total_neg += (masks == 0).sum().item()
    
    pos_weight_val = total_neg / (total_pos + 1e-6)
    # Cap pos_weight to prevent instability
    pos_weight_val = min(max(pos_weight_val, 1.0), 30.0)
    print(f"Computed pos_weight for BCE: {pos_weight_val:.2f}")

    # 2. Model, Loss, Optimizer
    model = TemporalUNet(in_channels=6, out_channels=1, init_features=32).to(DEVICE)
    
    # Custom loss: BCE with pos_weight + Dice
    pos_weight = torch.tensor([pos_weight_val]).to(DEVICE)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # We can reuse the Dice loss logic from CombinedBCEDiceLoss or write it
    class TemporalLoss(nn.Module):
        def __init__(self, bce, dice_weight=0.5):
            super().__init__()
            self.bce = bce
            self.dice_weight = dice_weight
            self.combined = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=dice_weight)
            # Override internal bce to use our weighted one
            self.combined.bce_loss = self.bce
            
        def forward(self, logits, targets):
            return self.combined(logits, targets)

    criterion = TemporalLoss(bce_loss_fn)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. History
    history = {
        "epoch": [], "train_loss": [], "train_iou": [], "train_dice": [],
        "val_loss": [], "val_iou": [], "val_dice": [], "val_precision": [], "val_recall": []
    }

    best_val_iou = -1.0
    Path("outputs").mkdir(exist_ok=True)
    out_csv = Path("outputs/temporal_training_history.csv")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_tp, train_fp, train_fn, train_tn = 0, 0, 0, 0

        for images, masks, _ in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            
            # metrics
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_tp += (preds * masks).sum().item()
            train_fp += (preds * (1 - masks)).sum().item()
            train_fn += ((1 - preds) * masks).sum().item()
            train_tn += ((1 - preds) * (1 - masks)).sum().item()

        train_loss /= len(train_loader.dataset)
        t_iou_den = train_tp + train_fp + train_fn
        train_iou = train_tp / t_iou_den if t_iou_den > 0 else 1.0
        t_dice_den = 2 * train_tp + train_fp + train_fn
        train_dice = (2 * train_tp) / t_dice_den if t_dice_den > 0 else 1.0

        model.eval()
        val_loss = 0.0
        val_tp, val_fp, val_fn, val_tn = 0, 0, 0, 0

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item() * images.size(0)

                preds = (torch.sigmoid(logits) > 0.5).float()
                val_tp += (preds * masks).sum().item()
                val_fp += (preds * (1 - masks)).sum().item()
                val_fn += ((1 - preds) * masks).sum().item()
                val_tn += ((1 - preds) * (1 - masks)).sum().item()

        val_loss /= len(val_loader.dataset)
        v_iou_den = val_tp + val_fp + val_fn
        val_iou = val_tp / v_iou_den if v_iou_den > 0 else 1.0
        v_dice_den = 2 * val_tp + val_fp + val_fn
        val_dice = (2 * val_tp) / v_dice_den if v_dice_den > 0 else 1.0
        v_prec_den = val_tp + val_fp
        val_prec = val_tp / v_prec_den if v_prec_den > 0 else 0.0
        v_rec_den = val_tp + val_fn
        val_rec = val_tp / v_rec_den if v_rec_den > 0 else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_iou"].append(train_iou)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
              f"TLoss: {train_loss:.4f} TIoU: {train_iou:.4f} TDice: {train_dice:.4f} | "
              f"VLoss: {val_loss:.4f} VIoU: {val_iou:.4f} VDice: {val_dice:.4f} "
              f"VPrec: {val_prec:.4f} VRec: {val_rec:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_iou": best_val_iou
            }, TEMPORAL_CHECKPOINT_PATH)
            print(f" -> Saved best model (IoU: {best_val_iou:.4f})")

    # Save History CSV
    with open(out_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        for i in range(len(history["epoch"])):
            writer.writerow([history[k][i] for k in history.keys()])

    # Plot Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.title("Loss vs Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["train_iou"], label="Train IoU")
    plt.plot(history["epoch"], history["val_iou"], label="Val IoU")
    plt.title("IoU vs Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig("outputs/temporal_training_curves.png")
    plt.close()

    print("Training Complete.")
    print(f"Best Validation IoU: {best_val_iou:.4f}")

if __name__ == "__main__":
    main()
