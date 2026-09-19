import sys
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure src directory is in sys.path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from temporal_dataset import get_temporal_dataloaders
from temporal_model import TemporalUNet, TEMPORAL_CHECKPOINT_PATH

DEVICE = torch.device("cpu")

def main():
    print("=" * 60)
    print("STAGE 2: TEMPORAL DEFORESTATION DETECTION EVALUATION")
    print("=" * 60)

    checkpoint_file = Path(TEMPORAL_CHECKPOINT_PATH)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Temporal model checkpoint not found: {TEMPORAL_CHECKPOINT_PATH}")

    # 1. Load Model
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        nk = k.replace("cu1.", "conv_up1.").replace("cu2.", "conv_up2.").replace("cu3.", "conv_up3.").replace("cu4.", "conv_up4.")
        new_state_dict[nk] = v

    model = TemporalUNet(in_channels=9, out_channels=1, init_features=32).to(DEVICE)
    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} with Val IoU: {checkpoint.get('best_val_iou', 0):.4f}")

    # 2. Dataloader
    _, _, test_loader = get_temporal_dataloaders(batch_size=1, diff_channels=True)
    print(f"Evaluating on Test set: {len(test_loader.dataset)} pairs")

    out_dir = Path("outputs/temporal")
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Global counters
    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_tn = 0
    total_pixels = 0

    with torch.no_grad():
        for i, (images, masks, names) in enumerate(test_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            # Global Metrics
            global_tp += (preds * masks).sum().item()
            global_fp += (preds * (1 - masks)).sum().item()
            global_fn += ((1 - preds) * masks).sum().item()
            global_tn += ((1 - preds) * (1 - masks)).sum().item()
            total_pixels += masks.numel()

            # Visualize first 5 examples
            if i < 5:
                name = names[0]
                # Reconstruct A and B from the 6-channel input
                img_a = images[0, 0:3].cpu().permute(1, 2, 0).numpy()
                img_b = images[0, 3:6].cpu().permute(1, 2, 0).numpy()
                gt_mask = masks[0, 0].cpu().numpy()
                pred_mask = preds[0, 0].cpu().numpy()

                fig = plt.figure(figsize=(15, 10))
                
                ax1 = fig.add_subplot(2, 3, 1)
                ax1.imshow(img_a)
                ax1.set_title("T1 (Before)")
                ax1.axis('off')
                
                ax2 = fig.add_subplot(2, 3, 2)
                ax2.imshow(img_b)
                ax2.set_title("T2 (After)")
                ax2.axis('off')
                
                ax3 = fig.add_subplot(2, 3, 3)
                ax3.imshow(gt_mask, cmap='Reds')
                ax3.set_title("Ground Truth Deforestation")
                ax3.axis('off')
                
                ax4 = fig.add_subplot(2, 3, 4)
                ax4.imshow(pred_mask, cmap='Reds')
                ax4.set_title("Predicted Deforestation")
                ax4.axis('off')
                
                ax5 = fig.add_subplot(2, 3, 5)
                ax5.imshow(img_b)
                overlay_pred = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
                overlay_pred[pred_mask == 1] = [1.0, 0.0, 0.0, 0.5]
                ax5.imshow(overlay_pred)
                ax5.set_title("Overlay Predicted on T2")
                ax5.axis('off')
                
                ax6 = fig.add_subplot(2, 3, 6)
                ax6.imshow(img_b)
                overlay_gt = np.zeros((*gt_mask.shape, 4), dtype=np.float32)
                overlay_gt[gt_mask == 1] = [1.0, 0.0, 0.0, 0.5]
                ax6.imshow(overlay_gt)
                ax6.set_title("Overlay Ground Truth on T2")
                ax6.axis('off')
                
                plt.tight_layout()
                plt.savefig(vis_dir / f"test_{name}_result.png", dpi=150)
                plt.close()

    # Calculate global metrics
    iou_denom = global_tp + global_fp + global_fn
    dice_denom = 2 * global_tp + global_fp + global_fn
    prec_denom = global_tp + global_fp
    rec_denom = global_tp + global_fn

    iou = global_tp / iou_denom if iou_denom > 0 else 1.0
    dice = (2 * global_tp) / dice_denom if dice_denom > 0 else 1.0
    precision = global_tp / prec_denom if prec_denom > 0 else 1.0
    recall = global_tp / rec_denom if rec_denom > 0 else 1.0
    accuracy = (global_tp + global_tn) / total_pixels if total_pixels > 0 else 1.0

    total_pred = global_tp + global_fp
    total_gt = global_tp + global_fn
    pred_pct = (total_pred / total_pixels) * 100 if total_pixels > 0 else 0
    gt_pct = (total_gt / total_pixels) * 100 if total_pixels > 0 else 0

    results = {
        "global_iou": float(iou),
        "global_dice": float(dice),
        "global_precision": float(precision),
        "global_recall": float(recall),
        "global_accuracy": float(accuracy),
        "tp": int(global_tp),
        "fp": int(global_fp),
        "fn": int(global_fn),
        "tn": int(global_tn),
        "total_predicted_deforestation_pixels": int(total_pred),
        "total_ground_truth_deforestation_pixels": int(total_gt),
        "predicted_deforestation_pct": float(pred_pct),
        "ground_truth_deforestation_pct": float(gt_pct)
    }

    with open(out_dir / "temporal_test_results.json", "w") as f:
        json.dump(results, f, indent=4)

    with open(out_dir / "temporal_test_report.txt", "w") as f:
        f.write("STAGE 2: TEMPORAL DEFORESTATION DETECTION TEST REPORT\n")
        f.write("===================================================\n")
        f.write(f"Evaluated on {len(test_loader.dataset)} Test Pairs\n\n")
        f.write(f"Global IoU: {iou:.4f}\n")
        f.write(f"Global Dice (F1): {dice:.4f}\n")
        f.write(f"Global Precision: {precision:.4f}\n")
        f.write(f"Global Recall: {recall:.4f}\n")
        f.write(f"Global Pixel Accuracy: {accuracy:.4f}\n\n")
        f.write(f"Total Predicted Deforestation: {total_pred:,} px ({pred_pct:.2f}%)\n")
        f.write(f"Total Ground Truth Deforestation: {total_gt:,} px ({gt_pct:.2f}%)\n")
        f.write(f"Confusion Counts -> TP: {int(global_tp)}, FP: {int(global_fp)}, FN: {int(global_fn)}, TN: {int(global_tn)}\n")

    print("\n--- TEST SET EVALUATION RESULTS ---")
    print(f"Global IoU       : {iou:.4f}")
    print(f"Global Dice      : {dice:.4f}")
    print(f"Global Precision : {precision:.4f}")
    print(f"Global Recall    : {recall:.4f}")
    print(f"Total Pred Defor : {total_pred:,} px ({pred_pct:.2f}%)")
    print(f"Total GT Defor   : {total_gt:,} px ({gt_pct:.2f}%)")
    print(f"\nSaved test report to {out_dir}/temporal_test_report.txt")
    print(f"Saved test visualizations to {vis_dir}")

if __name__ == "__main__":
    main()
