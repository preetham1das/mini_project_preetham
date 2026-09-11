import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# Ensure src directory is in sys.path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from temporal_dataset import get_temporal_dataloaders
from temporal_model import TemporalUNet, TEMPORAL_CHECKPOINT_PATH

DEVICE = torch.device("cpu")

def calculate_sample_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)

    iou_denom = tp + fp + fn
    dice_denom = 2 * tp + fp + fn

    iou = tp / iou_denom if iou_denom > 0 else (1.0 if np.sum(gt) == 0 and np.sum(pred) == 0 else 0.0)
    dice = (2 * tp) / dice_denom if dice_denom > 0 else (1.0 if np.sum(gt) == 0 and np.sum(pred) == 0 else 0.0)

    return float(iou), float(dice)

def main():
    print("=" * 60)
    print("STAGE 2: VISUALIZE BEST, MEDIAN, AND WORST SAMPLES")
    print("=" * 60)

    checkpoint_file = Path(TEMPORAL_CHECKPOINT_PATH)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Temporal model checkpoint not found: {TEMPORAL_CHECKPOINT_PATH}")

    # 1. Load Model
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE, weights_only=True)
    model = TemporalUNet(in_channels=6, out_channels=1, init_features=32).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Dataloader
    _, _, test_loader = get_temporal_dataloaders(batch_size=1)

    out_dir = Path("outputs/temporal/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    with torch.no_grad():
        for images, masks, names in test_loader:
            name = names[0]
            images = images.to(DEVICE)

            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()

            pred_mask = preds[0, 0].cpu().numpy()
            gt_mask = masks[0, 0].cpu().numpy()

            iou, dice = calculate_sample_metrics(pred_mask, gt_mask)

            # Reconstruct images for visualization
            img_a = images[0, 0:3].cpu().permute(1, 2, 0).numpy()
            img_b = images[0, 3:6].cpu().permute(1, 2, 0).numpy()

            results.append({
                "name": name,
                "iou": iou,
                "dice": dice,
                "img_a": img_a,
                "img_b": img_b,
                "gt_mask": gt_mask,
                "pred_mask": pred_mask
            })

    # Sort results by IoU
    results.sort(key=lambda x: x["iou"])

    worst_sample = results[0]
    median_sample = results[len(results) // 2]
    best_sample = results[-1]

    print(f"Worst Sample: {worst_sample['name']} | IoU: {worst_sample['iou']:.4f} | Dice: {worst_sample['dice']:.4f}")
    print(f"Median Sample: {median_sample['name']} | IoU: {median_sample['iou']:.4f} | Dice: {median_sample['dice']:.4f}")
    print(f"Best Sample: {best_sample['name']} | IoU: {best_sample['iou']:.4f} | Dice: {best_sample['dice']:.4f}")

    selected_samples = [
        ("Worst", worst_sample),
        ("Median", median_sample),
        ("Best", best_sample)
    ]

    # Plot individually
    for rank, sample in selected_samples:
        fig = plt.figure(figsize=(15, 4))
        plt.suptitle(f"{rank} Sample: {sample['name']} | IoU = {sample['iou']:.4f} | Dice = {sample['dice']:.4f}", fontsize=14)

        ax1 = fig.add_subplot(1, 5, 1)
        ax1.imshow(sample["img_a"])
        ax1.set_title("T1 (Before)")
        ax1.axis('off')

        ax2 = fig.add_subplot(1, 5, 2)
        ax2.imshow(sample["img_b"])
        ax2.set_title("T2 (After)")
        ax2.axis('off')

        ax3 = fig.add_subplot(1, 5, 3)
        ax3.imshow(sample["gt_mask"], cmap='Reds')
        ax3.set_title("GT Deforestation")
        ax3.axis('off')

        ax4 = fig.add_subplot(1, 5, 4)
        ax4.imshow(sample["pred_mask"], cmap='Reds')
        ax4.set_title("Pred Deforestation")
        ax4.axis('off')

        ax5 = fig.add_subplot(1, 5, 5)
        ax5.imshow(sample["img_b"])
        overlay = np.zeros((*sample["pred_mask"].shape, 4), dtype=np.float32)
        overlay[sample["pred_mask"] == 1] = [1.0, 0.0, 0.0, 0.5]
        ax5.imshow(overlay)
        ax5.set_title("Pred Overlay on T2")
        ax5.axis('off')

        plt.tight_layout()
        plt.savefig(out_dir / f"{rank.lower()}_{sample['name']}_visualization.png", dpi=150)
        plt.close()

    # Plot Combined Figure suitable for PPT
    fig = plt.figure(figsize=(15, 10))
    # plt.suptitle("Representative Temporal Deforestation Predictions", fontsize=16)

    for i, (rank, sample) in enumerate(selected_samples):
        # T1
        ax = fig.add_subplot(3, 5, i*5 + 1)
        ax.imshow(sample["img_a"])
        if i == 0: ax.set_title("T1 (Before)", fontweight='bold')
        ax.set_ylabel(f"{rank} (IoU: {sample['iou']:.2f})", rotation=90, size='large', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        # T2
        ax = fig.add_subplot(3, 5, i*5 + 2)
        ax.imshow(sample["img_b"])
        if i == 0: ax.set_title("T2 (After)", fontweight='bold')
        ax.axis('off')

        # GT
        ax = fig.add_subplot(3, 5, i*5 + 3)
        ax.imshow(sample["gt_mask"], cmap='Reds')
        if i == 0: ax.set_title("Ground Truth", fontweight='bold')
        ax.axis('off')

        # Pred
        ax = fig.add_subplot(3, 5, i*5 + 4)
        ax.imshow(sample["pred_mask"], cmap='Reds')
        if i == 0: ax.set_title("Prediction", fontweight='bold')
        ax.axis('off')

        # Overlay
        ax = fig.add_subplot(3, 5, i*5 + 5)
        ax.imshow(sample["img_b"])
        overlay = np.zeros((*sample["pred_mask"].shape, 4), dtype=np.float32)
        overlay[sample["pred_mask"] == 1] = [1.0, 0.0, 0.0, 0.5]
        ax.imshow(overlay)
        if i == 0: ax.set_title("Overlay", fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_dir / "combined_representative_results.png", dpi=300)
    plt.close()

    print(f"\nSaved representative visualizations to {out_dir}")

if __name__ == "__main__":
    main()
