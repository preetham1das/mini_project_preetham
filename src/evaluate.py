import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

# Ensure src directory is in sys.path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

try:
    from src.model import UNet
    from src.dataset import get_dataloaders
    from src.metrics import calculate_metrics, THRESHOLD
except ModuleNotFoundError:
    from model import UNet
    from dataset import get_dataloaders
    from metrics import calculate_metrics, THRESHOLD


def evaluate_model(
    checkpoint_path="outputs/best_unet.pth",
    dataset_root="data/Amazon Forest Dataset",
    output_dir="outputs",
    threshold=THRESHOLD
):
    print("=" * 60)
    print("MODEL VALIDATION EVALUATION & METRICS REPORTING")
    print("=" * 60)

    device = torch.device("cpu")
    checkpoint_file = Path(checkpoint_path)

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}. Train the model first.")

    # 1. Load Model
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_file, map_location=device)

    model = UNet(in_channels=3, out_channels=1, init_features=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"Classification Threshold: {threshold}")

    # 2. Load Validation Data
    _, val_loader, _ = get_dataloaders(dataset_root=dataset_root, batch_size=1)
    num_samples = len(val_loader.dataset)
    print(f"Evaluating {num_samples} validation samples...")

    all_metrics = []
    visualization_samples = []

    with torch.no_grad():
        for i, (image, mask) in enumerate(val_loader):
            image = image.to(device)
            mask = mask.to(device)

            logits = model(image)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            metrics = calculate_metrics(logits, mask, threshold=threshold, is_logits=True)
            all_metrics.append(metrics)

            if i < 4:  # Save first 4 samples for visual plotting
                img_np = image[0].permute(1, 2, 0).cpu().numpy()
                gt_np = mask[0].cpu().numpy()
                pred_np = preds[0, 0].cpu().numpy()
                prob_np = probs[0, 0].cpu().numpy()
                visualization_samples.append((img_np, gt_np, pred_np, prob_np))

    # 3. Compute Mean Metrics
    mean_metrics = {
        "iou": float(np.mean([m["iou"] for m in all_metrics])),
        "dice": float(np.mean([m["dice"] for m in all_metrics])),
        "precision": float(np.mean([m["precision"] for m in all_metrics])),
        "recall": float(np.mean([m["recall"] for m in all_metrics])),
        "accuracy": float(np.mean([m["accuracy"] for m in all_metrics])),
        "threshold": threshold,
        "total_samples": num_samples,
        "best_train_epoch": checkpoint.get("epoch", None)
    }

    print("\n--- VALIDATION PERFORMANCE SUMMARY ---")
    print(f"Mean IoU (Jaccard Index) : {mean_metrics['iou']:.4f} ({mean_metrics['iou']*100:.2f}%)")
    print(f"Mean Dice (F1 Score)     : {mean_metrics['dice']:.4f} ({mean_metrics['dice']*100:.2f}%)")
    print(f"Mean Precision           : {mean_metrics['precision']:.4f}")
    print(f"Mean Recall              : {mean_metrics['recall']:.4f}")
    print(f"Mean Pixel Accuracy      : {mean_metrics['accuracy']:.4f} ({mean_metrics['accuracy']*100:.2f}%)")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 4. Save JSON Report
    json_path = output_path / "evaluation_metrics.json"
    with open(json_path, "w") as f:
        json.dump(mean_metrics, f, indent=4)
    print(f"\nSaved metrics JSON report to: {json_path}")

    # 5. Save TXT Summary Report
    txt_path = output_path / "evaluation_report.txt"
    with open(txt_path, "w") as f:
        f.write("====================================================\n")
        f.write("DEFORESTATION DETECTION PROJECT - EVALUATION REPORT\n")
        f.write("====================================================\n\n")
        f.write(f"Checkpoint Loaded     : {checkpoint_path}\n")
        f.write(f"Checkpoint Epoch      : {checkpoint.get('epoch', 'N/A')}\n")
        f.write(f"Validation Samples    : {num_samples}\n")
        f.write(f"Prediction Threshold  : {threshold}\n\n")
        f.write("QUANTITATIVE METRICS:\n")
        f.write(f"- Mean IoU (Jaccard)   : {mean_metrics['iou']:.4f} ({mean_metrics['iou']*100:.2f}%)\n")
        f.write(f"- Mean Dice (F1 Score) : {mean_metrics['dice']:.4f} ({mean_metrics['dice']*100:.2f}%)\n")
        f.write(f"- Mean Precision       : {mean_metrics['precision']:.4f}\n")
        f.write(f"- Mean Recall          : {mean_metrics['recall']:.4f}\n")
        f.write(f"- Mean Pixel Accuracy  : {mean_metrics['accuracy']:.4f} ({mean_metrics['accuracy']*100:.2f}%)\n")
    print(f"Saved text evaluation report to: {txt_path}")

    # 6. Generate Visualization Grid
    fig, axes = plt.subplots(len(visualization_samples), 4, figsize=(16, 4 * len(visualization_samples)))
    if len(visualization_samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (img_np, gt_np, pred_np, prob_np) in enumerate(visualization_samples):
        # Image
        axes[idx, 0].imshow(np.clip(img_np, 0, 1))
        axes[idx, 0].set_title(f"Sample {idx+1}: Satellite RGB")
        axes[idx, 0].axis("off")

        # Ground Truth Mask (1 = Forest, 0 = Non-forest)
        axes[idx, 1].imshow(gt_np, cmap="Greens", vmin=0, vmax=1)
        axes[idx, 1].set_title(f"Ground Truth Forest Mask")
        axes[idx, 1].axis("off")

        # Predicted Forest Mask
        axes[idx, 2].imshow(pred_np, cmap="Greens", vmin=0, vmax=1)
        axes[idx, 2].set_title(f"U-Net Forest Prediction (> {threshold})")
        axes[idx, 2].axis("off")

        # Error Map: Green = TP, Red = FP/FN, Dark Gray = TN
        error_map = np.zeros((*gt_np.shape, 3), dtype=np.float32)
        tp_mask = (pred_np == 1) & (gt_np == 1)
        error_mask = (pred_np != gt_np)
        tn_mask = (pred_np == 0) & (gt_np == 0)

        error_map[tp_mask] = [0.0, 0.8, 0.0]     # Green (Correct Forest)
        error_map[error_mask] = [0.9, 0.1, 0.1]  # Red (Mismatch)
        error_map[tn_mask] = [0.2, 0.2, 0.2]     # Dark Gray (Correct Non-Forest)

        axes[idx, 3].imshow(error_map)
        axes[idx, 3].set_title("Error Map (Green=TP, Red=Err)")
        axes[idx, 3].axis("off")

    plt.tight_layout()
    visuals_path = output_path / "val_evaluation_visuals.png"
    plt.savefig(visuals_path, dpi=150)
    plt.close()

    print(f"Saved validation evaluation visuals to: {visuals_path}")
    print("=" * 60)

    return mean_metrics


if __name__ == "__main__":
    evaluate_model()
