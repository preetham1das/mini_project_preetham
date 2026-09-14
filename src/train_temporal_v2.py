"""
train_temporal_v2.py
====================
Unified training script for Stage 2 temporal deforestation detection.

Runs experiments A, B, C (and optionally D) sequentially.
Each experiment saves:
  - outputs/experiments/<exp_name>/training_history.csv
  - outputs/experiments/<exp_name>/training_curves.png
  - outputs/experiments/<exp_name>/best_model.pth

After all experiments, the one with the highest best_val_iou is copied to:
  - models/best_temporal_model.pth

Usage:
  python src/train_temporal_v2.py

Experiments:
  A  --  6-ch input, WeightedBCE+Dice, 30 epochs w/ early stopping
  B  --  9-ch input (|A-B|), WeightedBCE+Dice, 30 epochs
  C  --  9-ch input, Focal+Dice, 30 epochs
  D  --  9-ch input, Focal+Dice, init_features=48 (larger model, optional)
"""

import sys
import csv
import shutil
import random
import argparse
from pathlib import Path

import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from temporal_dataset import get_temporal_dataloaders, compute_pos_weight
from temporal_model import (
    TemporalUNet,
    WeightedBCEDiceLoss,
    FocalDiceLoss,
    TEMPORAL_CHECKPOINT_PATH,
)

# ------------------------------------------------------------------------------
# Global config
# ------------------------------------------------------------------------------
SEED       = 42
BATCH_SIZE = 4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 30
PATIENCE   = 8    # early-stopping patience (epochs without val IoU improvement)
LR         = 1e-3

DATASET_ROOT = "data/Forest-Change/extracted/Forest-Change-dataset/images"
EXP_DIR      = Path("outputs/experiments")
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Experiment definitions
# ------------------------------------------------------------------------------
EXPERIMENTS = {
    "A": dict(
        diff_channels=False,
        in_channels=6,
        init_features=32,
        bottleneck_dropout=0.0,
        loss="wbce_dice",
        description="6-ch [A,B] | WeightedBCE+Dice | lr=1e-3->RLP | 30ep/p=8",
    ),
    "B": dict(
        diff_channels=True,
        in_channels=9,
        init_features=32,
        bottleneck_dropout=0.0,
        loss="wbce_dice",
        description="9-ch [A,B,|A-B|] | WeightedBCE+Dice | lr=1e-3->RLP | 30ep/p=8",
    ),
    "C": dict(
        diff_channels=True,
        in_channels=9,
        init_features=32,
        bottleneck_dropout=0.0,
        loss="focal_dice",
        description="9-ch [A,B,|A-B|] | Focal(g=2)+Dice | lr=1e-3->RLP | 30ep/p=8",
    ),
    "D": dict(
        diff_channels=True,
        in_channels=9,
        init_features=48,
        bottleneck_dropout=0.1,
        loss="focal_dice",
        description="9-ch [A,B,|A-B|] | Focal+Dice | f=48 | dropout=0.1 | 40ep/p=10",
        max_epochs=40,
        patience=10,
    ),
}


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def iou_from_counts(tp, fp, fn):
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 1.0


def dice_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom > 0 else 1.0


def eval_epoch(model, loader, criterion, threshold=0.5):
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for images, masks, _ in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(logits) > threshold).float()
            tp += (preds * masks).sum().item()
            fp += (preds * (1 - masks)).sum().item()
            fn += ((1 - preds) * masks).sum().item()
            tn += ((1 - preds) * (1 - masks)).sum().item()

    n = len(loader.dataset)
    return dict(
        loss=total_loss / n,
        iou=iou_from_counts(tp, fp, fn),
        dice=dice_from_counts(tp, fp, fn),
        precision=tp / (tp + fp) if (tp + fp) > 0 else 1.0,
        recall=tp / (tp + fn) if (tp + fn) > 0 else 1.0,
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


def plot_curves(history, out_path):
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].plot(epochs, history["train_iou"],  label="Train")
    axes[1].plot(epochs, history["val_iou"],    label="Val")
    axes[1].set_title("IoU"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


# ------------------------------------------------------------------------------
# Single experiment runner
# ------------------------------------------------------------------------------
def run_experiment(name, cfg, pos_weight_val, pos_weight_val_6ch=None):
    print("\n" + "=" * 65)
    print(f"EXPERIMENT {name}: {cfg['description']}")
    print("=" * 65)

    set_seed(SEED)

    diff = cfg["diff_channels"]
    max_ep = cfg.get("max_epochs", MAX_EPOCHS)
    patience = cfg.get("patience", PATIENCE)

    # Dataloaders
    train_loader, val_loader, _ = get_temporal_dataloaders(
        dataset_root=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        diff_channels=diff,
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # Model
    model = TemporalUNet(
        in_channels=cfg["in_channels"],
        out_channels=1,
        init_features=cfg["init_features"],
        bottleneck_dropout=cfg["bottleneck_dropout"],
    ).to(DEVICE)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {params:,}")

    # Loss
    if cfg["loss"] == "wbce_dice":
        criterion = WeightedBCEDiceLoss(pos_weight_val=pos_weight_val, bce_weight=0.5, dice_weight=0.5)
        print(f"Loss: WeightedBCE+Dice (pos_weight={pos_weight_val:.2f})")
    elif cfg["loss"] == "focal_dice":
        criterion = FocalDiceLoss(focal_weight=0.5, dice_weight=0.5, gamma=2.0, alpha=0.75)
        print("Loss: Focal(g=2,a=0.75)+Dice")
    else:
        raise ValueError(f"Unknown loss: {cfg['loss']}")

    # Optimizer + scheduler
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-5
    )

    # Output dir
    out_dir = EXP_DIR / f"exp_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {k: [] for k in ["epoch","train_loss","train_iou","train_dice",
                                "val_loss","val_iou","val_dice","val_precision","val_recall","lr"]}

    best_val_iou = -1.0
    epochs_no_improve = 0
    ckpt_path = out_dir / "best_model.pth"

    for epoch in range(1, max_ep + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        tp = fp = fn = tn = 0

        for images, masks, _ in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            tp += (preds * masks).sum().item()
            fp += (preds * (1 - masks)).sum().item()
            fn += ((1 - preds) * masks).sum().item()
            tn += ((1 - preds) * (1 - masks)).sum().item()

        train_loss /= len(train_loader.dataset)
        train_iou  = iou_from_counts(tp, fp, fn)
        train_dice = dice_from_counts(tp, fp, fn)

        # -- Validate --
        vm = eval_epoch(model, val_loader, criterion)

        # Scheduler steps on val IoU
        scheduler.step(vm["iou"])
        current_lr = optimizer.param_groups[0]["lr"]

        # -- Log --
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_iou"].append(train_iou)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(vm["loss"])
        history["val_iou"].append(vm["iou"])
        history["val_dice"].append(vm["dice"])
        history["val_precision"].append(vm["precision"])
        history["val_recall"].append(vm["recall"])
        history["lr"].append(current_lr)

        print(
            f"Ep [{epoch:02d}/{max_ep}] "
            f"TLoss={train_loss:.4f} TIoU={train_iou:.4f} | "
            f"VLoss={vm['loss']:.4f} VIoU={vm['iou']:.4f} VDice={vm['dice']:.4f} "
            f"VPrec={vm['precision']:.4f} VRec={vm['recall']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # -- Save best --
        if vm["iou"] > best_val_iou:
            best_val_iou = vm["iou"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_iou": best_val_iou,
                "config": cfg,
            }, ckpt_path)
            print(f"   -> Saved best model (Val IoU={best_val_iou:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"   -> Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Save CSV
    csv_path = out_dir / "training_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        for i in range(len(history["epoch"])):
            writer.writerow([history[k][i] for k in history.keys()])

    plot_curves(history, out_dir / "training_curves.png")

    print(f"\nExperiment {name} done. Best Val IoU = {best_val_iou:.4f}")
    return best_val_iou, ckpt_path


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exps", nargs="+", default=["A", "B", "C"],
        choices=["A", "B", "C", "D"],
        help="Which experiments to run (default: A B C)"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("STAGE 2 v2: TEMPORAL DEFORESTATION DETECTION")
    print("=" * 65)
    print(f"Running experiments: {args.exps}")

    # Compute exact pos_weight from full training set ONCE
    print("\nComputing pos_weight from full training set...")
    pos_weight_val = compute_pos_weight(DATASET_ROOT)
    print(f"  True neg/pos ratio = {pos_weight_val:.2f}")

    # Run each experiment
    results = {}
    for name in args.exps:
        if name not in EXPERIMENTS:
            print(f"Unknown experiment: {name}, skipping.")
            continue
        cfg = EXPERIMENTS[name]
        val_iou, ckpt = run_experiment(name, cfg, pos_weight_val)
        results[name] = {"val_iou": val_iou, "ckpt": ckpt}

    # Summary
    print("\n" + "=" * 65)
    print("EXPERIMENT SUMMARY")
    print("=" * 65)
    best_name = None
    best_iou  = -1.0
    for name, r in results.items():
        desc = EXPERIMENTS[name]["description"]
        print(f"  Exp {name}: Val IoU = {r['val_iou']:.4f}  [{desc}]")
        if r["val_iou"] > best_iou:
            best_iou  = r["val_iou"]
            best_name = name

    print(f"\nBest experiment: {best_name} (Val IoU = {best_iou:.4f})")

    # Copy best checkpoint to canonical location
    best_ckpt = results[best_name]["ckpt"]
    Path("models").mkdir(exist_ok=True)
    shutil.copy2(best_ckpt, TEMPORAL_CHECKPOINT_PATH)
    print(f"Copied best checkpoint to {TEMPORAL_CHECKPOINT_PATH}")

    # Save summary
    summary_path = EXP_DIR / "experiment_summary.txt"
    with open(summary_path, "w") as f:
        f.write("STAGE 2 v2 EXPERIMENT SUMMARY\n")
        f.write("=" * 65 + "\n\n")
        for name, r in results.items():
            cfg = EXPERIMENTS[name]
            f.write(f"Exp {name}: {cfg['description']}\n")
            f.write(f"  Val IoU = {r['val_iou']:.4f}\n\n")
        f.write(f"Best: Exp {best_name}  Val IoU = {best_iou:.4f}\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
