import sys
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Ensure src directory is in sys.path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

try:
    from src.model import CHECKPOINT_PATH, UNet
    from src.metrics import THRESHOLD
except ModuleNotFoundError:
    from model import CHECKPOINT_PATH, UNet
    from metrics import THRESHOLD

TARGET_SIZE = (256, 256)
DATASET_ROOT = Path("data/Forest-Change/extracted/Forest-Change-dataset/images")

def load_and_preprocess_image(image_path, target_size=TARGET_SIZE, is_label=False):
    """
    Loads image, resizes, and returns numpy array. For model input, returns tensor too.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    if is_label:
        pil_img = Image.open(path).convert("L")
        if pil_img.size != target_size:
            pil_img = pil_img.resize(target_size, Image.Resampling.NEAREST)
        img_np = np.array(pil_img, dtype=np.uint8)
        return img_np
    else:
        pil_img = Image.open(path).convert("RGB")
        if pil_img.size != target_size:
            pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return img_tensor, img_np

def predict_forest_mask(model, image_tensor, device, threshold=THRESHOLD):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        mask = preds[0, 0].cpu().numpy().astype(np.uint8)
    return mask

def calculate_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    tn = np.sum(~pred & ~gt)

    iou_denom = tp + fp + fn
    dice_denom = 2 * tp + fp + fn
    prec_denom = tp + fp
    rec_denom = tp + fn
    acc_denom = tp + fp + fn + tn

    iou = tp / iou_denom if iou_denom > 0 else 1.0
    dice = (2 * tp) / dice_denom if dice_denom > 0 else 1.0
    precision = tp / prec_denom if prec_denom > 0 else 0.0
    recall = tp / rec_denom if rec_denom > 0 else 0.0
    accuracy = (tp + tn) / acc_denom if acc_denom > 0 else 1.0

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }

def detect_deforestation(
    split,
    image_id,
    checkpoint_path=CHECKPOINT_PATH,
    output_dir="outputs/temporal",
    threshold=THRESHOLD
):
    print("=" * 70)
    print("STAGE 2: TEMPORAL DEFORESTATION DETECTION")
    print("=" * 70)

    device = torch.device("cpu")
    checkpoint_file = Path(checkpoint_path)

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    # Load Model
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
    model = UNet(in_channels=3, out_channels=1, init_features=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Paths
    t1_path = DATASET_ROOT / split / "A" / f"{image_id}.png"
    t2_path = DATASET_ROOT / split / "B" / f"{image_id}.png"
    label_path = DATASET_ROOT / split / "label" / f"{image_id}.png"

    for p in [t1_path, t2_path, label_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required dataset file: {p}")

    print(f"Time 1 (A) : {t1_path}")
    print(f"Time 2 (B) : {t2_path}")
    print(f"Label      : {label_path}")

    # Process Images
    t1_tensor, t1_img_np = load_and_preprocess_image(t1_path, TARGET_SIZE, False)
    t2_tensor, t2_img_np = load_and_preprocess_image(t2_path, TARGET_SIZE, False)
    label_np = load_and_preprocess_image(label_path, TARGET_SIZE, True)

    # Convert Label to Binary GT Change Mask (Forest-Change uses > 0 for change)
    gt_change_mask = (label_np > 0).astype(np.uint8)

    # Predictions
    t1_mask = predict_forest_mask(model, t1_tensor, device, threshold)
    t2_mask = predict_forest_mask(model, t2_tensor, device, threshold)

    # Compute Temporal Maps
    forest_loss_mask = ((t1_mask == 1) & (t2_mask == 0)).astype(np.uint8)
    forest_gain_mask = ((t1_mask == 0) & (t2_mask == 1)).astype(np.uint8)
    stable_forest_mask = ((t1_mask == 1) & (t2_mask == 1)).astype(np.uint8)
    stable_nonforest_mask = ((t1_mask == 0) & (t2_mask == 0)).astype(np.uint8)

    # Metrics computation for Deforestation (Forest Loss)
    metrics = calculate_metrics(forest_loss_mask, gt_change_mask)

    total_pixels = TARGET_SIZE[0] * TARGET_SIZE[1]

    t1_forest_pixels = int(np.sum(t1_mask == 1))
    t2_forest_pixels = int(np.sum(t2_mask == 1))

    pred_loss_pixels = int(np.sum(forest_loss_mask))
    gt_loss_pixels = int(np.sum(gt_change_mask))
    pred_gain_pixels = int(np.sum(forest_gain_mask))
    pred_stable_forest_pixels = int(np.sum(stable_forest_mask))
    pred_stable_nonforest_pixels = int(np.sum(stable_nonforest_mask))

    t1_forest_pct = (t1_forest_pixels / total_pixels) * 100
    t2_forest_pct = (t2_forest_pixels / total_pixels) * 100
    pred_loss_pct = (pred_loss_pixels / total_pixels) * 100
    gt_loss_pct = (gt_loss_pixels / total_pixels) * 100
    pred_gain_pct = (pred_gain_pixels / total_pixels) * 100
    pred_stable_forest_pct = (pred_stable_forest_pixels / total_pixels) * 100
    pred_stable_nonforest_pct = (pred_stable_nonforest_pixels / total_pixels) * 100

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    vis_path = out_path / "visualizations"
    vis_path.mkdir(parents=True, exist_ok=True)

    # Save Report
    report_path = out_path / "temporal_report.txt"
    with open(report_path, "w") as f:
        f.write("TEMPORAL DEFORESTATION DETECTION REPORT\n")
        f.write("=======================================\n")
        f.write(f"Pair ID: {split}/{image_id}\n\n")
        f.write(f"T1 Predicted Forest: {t1_forest_pct:.2f}% ({t1_forest_pixels} px)\n")
        f.write(f"T2 Predicted Forest: {t2_forest_pct:.2f}% ({t2_forest_pixels} px)\n")
        f.write(f"Predicted Deforestation (Loss): {pred_loss_pct:.2f}% ({pred_loss_pixels} px)\n")
        f.write(f"Predicted Forest Gain: {pred_gain_pct:.2f}% ({pred_gain_pixels} px)\n")
        f.write(f"Predicted Stable Forest: {pred_stable_forest_pct:.2f}% ({pred_stable_forest_pixels} px)\n")
        f.write(f"Predicted Stable Non-Forest: {pred_stable_nonforest_pct:.2f}% ({pred_stable_nonforest_pixels} px)\n")
        f.write(f"Ground-Truth Deforestation: {gt_loss_pct:.2f}% ({gt_loss_pixels} px)\n\n")
        f.write("Deforestation Evaluation Metrics:\n")
        for k, v in metrics.items():
            f.write(f"- {k.capitalize()}: {v:.4f}\n" if isinstance(v, float) else f"- {k.capitalize()}: {v}\n")

    json_path = out_path / "temporal_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "t1_forest_pct": t1_forest_pct,
            "t2_forest_pct": t2_forest_pct,
            "pred_loss_pct": pred_loss_pct,
            "gt_loss_pct": gt_loss_pct,
            "pred_loss_pixels": pred_loss_pixels,
            "gt_loss_pixels": gt_loss_pixels,
            "pred_gain_pixels": pred_gain_pixels,
            "pred_stable_forest_pixels": pred_stable_forest_pixels,
            "pred_stable_nonforest_pixels": pred_stable_nonforest_pixels,
            "metrics": metrics
        }, f, indent=4)

    print(f"\nPredicted Deforestation: {pred_loss_pixels} px ({pred_loss_pct:.2f}%)")
    print(f"Ground-Truth Deforestation: {gt_loss_pixels} px ({gt_loss_pct:.2f}%)")
    print(f"IoU: {metrics['iou']:.4f} | Dice: {metrics['dice']:.4f}")

    # Visualizations
    fig = plt.figure(figsize=(18, 12))

    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(t1_img_np)
    ax1.set_title("T1 Image (Before)")
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(t2_img_np)
    ax2.set_title("T2 Image (After)")
    ax2.axis('off')

    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(t1_mask, cmap='gray')
    ax3.set_title("T1 Predicted Forest")
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 4, 4)
    ax4.imshow(t2_mask, cmap='gray')
    ax4.set_title("T2 Predicted Forest")
    ax4.axis('off')

    ax5 = fig.add_subplot(2, 4, 5)
    ax5.imshow(forest_loss_mask, cmap='Reds')
    ax5.set_title("Predicted Deforestation")
    ax5.axis('off')

    ax6 = fig.add_subplot(2, 4, 6)
    ax6.imshow(gt_change_mask, cmap='Reds')
    ax6.set_title("Ground-Truth Deforestation")
    ax6.axis('off')

    # Overlay predicted deforestation on T2
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.imshow(t2_img_np)
    # create a red mask with alpha for deforestation
    overlay = np.zeros((*TARGET_SIZE, 4), dtype=np.float32)
    overlay[forest_loss_mask == 1] = [1.0, 0.0, 0.0, 0.5] # Red with 50% opacity
    ax7.imshow(overlay)
    ax7.set_title("Deforestation Overlay on T2")
    ax7.axis('off')

    plt.tight_layout()
    fig_path = vis_path / f"{split}_{image_id}_summary.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\nResults saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Temporal Deforestation Detection")
    parser.add_argument("--split", type=str, help="Dataset split (train, val, test)", required=False)
    parser.add_argument("--id", type=str, help="Image ID (e.g., test_000001)", required=False)

    args = parser.parse_args()

    if not args.split or not args.id:
        print("No pair specified. Defaulting to a known pair for demonstration.")
        print("Usage: python src/deforestation_detection.py --split test --id test_000001")
        split = "test"
        image_id = "test_000001"
    else:
        split = args.split
        image_id = args.id

    detect_deforestation(split, image_id)

if __name__ == "__main__":
    main()
