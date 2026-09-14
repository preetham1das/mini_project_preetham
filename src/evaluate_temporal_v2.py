"""
evaluate_temporal_v2.py
========================
Evaluation script for Stage 2 temporal deforestation detection.

Steps:
  1. Load the best checkpoint (models/best_temporal_model.pth)
  2. Run threshold sweep [0.2, 0.3, 0.4, 0.5, 0.6] on VALIDATION set only
  3. Select the threshold with the highest val IoU
  4. Evaluate the TEST set ONCE at the selected threshold
  5. Save visualizations (best / median / worst test samples)
  6. Print a full report

Usage:
  python src/evaluate_temporal_v2.py
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from temporal_dataset import get_temporal_dataloaders, TemporalForestDataset
from temporal_model import TemporalUNet, TEMPORAL_CHECKPOINT_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = "data/Forest-Change/extracted/Forest-Change-dataset/images"
OUT_DIR = Path("outputs/temporal_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR = OUT_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)


#                                                                               
def iou_from_counts(tp, fp, fn):
    d = tp + fp + fn
    return tp / d if d > 0 else 1.0


def confusion_at_threshold(model, loader, threshold):
    """Return global TP/FP/FN/TN at a given threshold."""
    tp = fp = fn = tn = 0
    model.eval()
    with torch.no_grad():
        for images, masks, _ in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            tp += (preds * masks).sum().item()
            fp += (preds * (1 - masks)).sum().item()
            fn += ((1 - preds) * masks).sum().item()
            tn += ((1 - preds) * (1 - masks)).sum().item()
    return tp, fp, fn, tn


def full_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    return dict(
        iou=iou_from_counts(tp, fp, fn),
        dice=(2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0,
        precision=tp / (tp + fp) if (tp + fp) > 0 else 1.0,
        recall=tp / (tp + fn) if (tp + fn) > 0 else 1.0,
        accuracy=(tp + tn) / total if total > 0 else 1.0,
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    )


#                                                                               
def main():
    print("=" * 65)
    print("STAGE 2 v2: TEMPORAL DEFORESTATION EVALUATION")
    print("=" * 65)

    # 1. Load checkpoint                                                       
    ckpt_path = Path(TEMPORAL_CHECKPOINT_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = checkpoint.get("config", {})

    in_channels        = cfg.get("in_channels", 6)
    init_features      = cfg.get("init_features", 32)
    bottleneck_dropout = cfg.get("bottleneck_dropout", 0.0)
    diff_channels      = cfg.get("diff_channels", False)

    model = TemporalUNet(
        in_channels=in_channels,
        out_channels=1,
        init_features=init_features,
        bottleneck_dropout=bottleneck_dropout,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_val_iou_at_save = checkpoint.get("best_val_iou", 0.0)
    print(f"Loaded checkpoint   epoch {checkpoint.get('epoch','?')}, "
          f"Val IoU at save = {best_val_iou_at_save:.4f}")
    print(f"Model config: in_channels={in_channels}, init_features={init_features}, "
          f"diff_channels={diff_channels}")

    # 2. Dataloaders                                                           
    _, val_loader, test_loader = get_temporal_dataloaders(
        dataset_root=DATASET_ROOT,
        batch_size=1,
        diff_channels=diff_channels,
    )
    print(f"Val: {len(val_loader.dataset)} pairs | Test: {len(test_loader.dataset)} pairs")

    # 3. Threshold sweep on VALIDATION set                                     
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print("\nThreshold sweep on VALIDATION set:")
    print(f"  {'Threshold':<12} {'Val IoU':<10} {'Val Dice':<10} {'Val Prec':<10} {'Val Rec':<10}")

    best_thresh = 0.5
    best_thresh_iou = -1.0
    thresh_results = []

    for t in thresholds:
        tp, fp, fn, tn = confusion_at_threshold(model, val_loader, t)
        m = full_metrics(tp, fp, fn, tn)
        print(f"  {t:<12.1f} {m['iou']:<10.4f} {m['dice']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f}")
        thresh_results.append({"threshold": t, **m})
        if m["iou"] > best_thresh_iou:
            best_thresh_iou = m["iou"]
            best_thresh = t

    print(f"\nBest validation threshold: {best_thresh} (Val IoU = {best_thresh_iou:.4f})")

    # 4. Final TEST evaluation at best threshold                                
    print(f"\nFinal TEST evaluation at threshold = {best_thresh}:")
    tp, fp, fn, tn = confusion_at_threshold(model, test_loader, best_thresh)
    test_m = full_metrics(tp, fp, fn, tn)

    total_pred = tp + fp
    total_gt   = tp + fn
    total_px   = tp + fp + fn + tn

    print(f"  Test IoU       : {test_m['iou']:.4f}")
    print(f"  Test Dice      : {test_m['dice']:.4f}")
    print(f"  Test Precision : {test_m['precision']:.4f}")
    print(f"  Test Recall    : {test_m['recall']:.4f}")
    print(f"  Test Accuracy  : {test_m['accuracy']:.4f}")
    print(f"  Predicted deforestation: {int(total_pred):,} px ({total_pred/total_px*100:.2f}%)")
    print(f"  Ground-truth deforestation: {int(total_gt):,} px ({total_gt/total_px*100:.2f}%)")

    # 5. Per-sample IoU for visualization selection                            
    per_sample = []
    split_root = Path(DATASET_ROOT) / "test"
    ds = TemporalForestDataset(split_root, augment=False, diff_channels=diff_channels)

    model.eval()
    with torch.no_grad():
        for images, masks, names in test_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > best_thresh).float()
            tp_i = (preds * masks).sum().item()
            fp_i = (preds * (1 - masks)).sum().item()
            fn_i = ((1 - preds) * masks).sum().item()
            iou_i = iou_from_counts(tp_i, fp_i, fn_i)
            per_sample.append({
                "name": names[0],
                "iou": iou_i,
                "images": images.cpu(),
                "masks": masks.cpu(),
                "preds": preds.cpu(),
            })

    per_sample.sort(key=lambda x: x["iou"])
    best_s   = per_sample[-1]
    worst_s  = per_sample[0]
    median_s = per_sample[len(per_sample) // 2]

    print(f"\nTest sample range: worst IoU={worst_s['iou']:.4f}, "
          f"median IoU={median_s['iou']:.4f}, best IoU={best_s['iou']:.4f}")

    # 6. Visualizations                                                        
    def save_vis(sample, label, filename):
        images = sample["images"]
        masks  = sample["masks"]
        preds  = sample["preds"]

        img_a = images[0, 0:3].permute(1, 2, 0).numpy()
        img_b = images[0, 3:6].permute(1, 2, 0).numpy()
        gt    = masks[0, 0].numpy()
        pred  = preds[0, 0].numpy()

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        axes[0].imshow(img_a); axes[0].set_title("T1 (Before)"); axes[0].axis("off")
        axes[1].imshow(img_b); axes[1].set_title("T2 (After)");  axes[1].axis("off")
        axes[2].imshow(gt,   cmap="Reds"); axes[2].set_title("Ground Truth"); axes[2].axis("off")
        axes[3].imshow(pred, cmap="Reds"); axes[3].set_title(f"Prediction (IoU={sample['iou']:.3f})"); axes[3].axis("off")

        plt.suptitle(f"{label}   {sample['name']}", fontsize=13)
        plt.tight_layout()
        plt.savefig(VIS_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close()

    save_vis(best_s,   "Best Test Sample",   "best_sample.png")
    save_vis(median_s, "Median Test Sample", "median_sample.png")
    save_vis(worst_s,  "Worst Test Sample",  "worst_sample.png")

    # Combined 3-row figure
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for row_i, (sample, label) in enumerate([
        (best_s, f"Best (IoU={best_s['iou']:.3f})"),
        (median_s, f"Median (IoU={median_s['iou']:.3f})"),
        (worst_s, f"Worst (IoU={worst_s['iou']:.3f})"),
    ]):
        images = sample["images"]
        masks  = sample["masks"]
        preds  = sample["preds"]
        img_a = images[0, 0:3].permute(1, 2, 0).numpy()
        img_b = images[0, 3:6].permute(1, 2, 0).numpy()
        gt    = masks[0, 0].numpy()
        pred  = preds[0, 0].numpy()
        axes[row_i, 0].imshow(img_a);  axes[row_i, 0].set_title("T1 (Before)"); axes[row_i, 0].axis("off")
        axes[row_i, 1].imshow(img_b);  axes[row_i, 1].set_title("T2 (After)");  axes[row_i, 1].axis("off")
        axes[row_i, 2].imshow(gt,   cmap="Reds"); axes[row_i, 2].set_title("Ground Truth"); axes[row_i, 2].axis("off")
        axes[row_i, 3].imshow(pred, cmap="Reds"); axes[row_i, 3].set_title(f"Pred"); axes[row_i, 3].axis("off")
        axes[row_i, 0].set_ylabel(label, fontsize=11)

    plt.suptitle(f"Stage 2 Test Results (threshold={best_thresh})", fontsize=14)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "combined_results.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 7. Save JSON and report                                                  
    output = {
        "checkpoint": str(ckpt_path),
        "model_config": {
            "in_channels": in_channels,
            "init_features": init_features,
            "diff_channels": diff_channels,
        },
        "val_iou_at_save": float(best_val_iou_at_save),
        "threshold_sweep": thresh_results,
        "best_threshold": best_thresh,
        "best_val_iou_at_best_threshold": float(best_thresh_iou),
        "test_metrics": {k: float(v) if isinstance(v, float) else v for k, v in test_m.items()},
        "test_predicted_deforestation_pct": float(total_pred / total_px * 100),
        "test_ground_truth_deforestation_pct": float(total_gt / total_px * 100),
    }

    with open(OUT_DIR / "temporal_v2_results.json", "w") as f:
        json.dump(output, f, indent=4)

    report_lines = [
        "STAGE 2 v2: TEMPORAL DEFORESTATION DETECTION   FINAL REPORT",
        "=" * 65,
        f"Checkpoint: {ckpt_path}",
        f"Model: in_channels={in_channels}, init_features={init_features}, diff_channels={diff_channels}",
        "",
        "THRESHOLD SWEEP (Validation Set):",
        f"  {'Threshold':<12} {'Val IoU':<10} {'Val Dice':<10}",
    ]
    for r in thresh_results:
        report_lines.append(f"  {r['threshold']:<12.1f} {r['iou']:<10.4f} {r['dice']:<10.4f}")
    report_lines += [
        "",
        f"Selected threshold: {best_thresh}  (Best Val IoU = {best_thresh_iou:.4f})",
        "",
        "FINAL TEST METRICS:",
        f"  IoU       : {test_m['iou']:.4f}",
        f"  Dice (F1) : {test_m['dice']:.4f}",
        f"  Precision : {test_m['precision']:.4f}",
        f"  Recall    : {test_m['recall']:.4f}",
        f"  Accuracy  : {test_m['accuracy']:.4f}",
        f"  TP={int(tp)}, FP={int(fp)}, FN={int(fn)}, TN={int(tn)}",
    ]

    report_txt = "\n".join(report_lines)
    with open(OUT_DIR / "temporal_v2_report.txt", "w") as f:
        f.write(report_txt)

    print(f"\nResults saved to {OUT_DIR}/")
    print(f"Visualizations saved to {VIS_DIR}/")


if __name__ == "__main__":
    main()
