import sys
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
    from src.model import UNet
    from src.metrics import THRESHOLD
except ModuleNotFoundError:
    from model import UNet
    from metrics import THRESHOLD

TARGET_SIZE = (512, 512)


def load_and_preprocess_image(image_path, target_size=TARGET_SIZE):
    """
    Loads satellite image, resizes to target_size, and returns normalized tensor and numpy array.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    pil_img = Image.open(path).convert("RGB")
    if pil_img.size != target_size:
        pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)

    img_np = np.array(pil_img, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img_tensor, img_np


def predict_forest_mask(model, image_tensor, device, threshold=THRESHOLD):
    """
    Predicts binary forest mask (1 = Forest, 0 = Non-Forest) using U-Net model and threshold.
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        mask = preds[0, 0].cpu().numpy().astype(np.uint8)
    return mask, probs[0, 0].cpu().numpy()


def detect_deforestation(
    t1_image_path,
    t2_image_path,
    checkpoint_path="outputs/best_unet.pth",
    output_dir="outputs",
    threshold=THRESHOLD
):
    """
    Performs temporal deforestation change detection between Time 1 (Before) and Time 2 (After)
    satellite images of the SAME geographical region.
    """
    print("=" * 70)
    print("TEMPORAL DEFORESTATION & FOREST COVER CHANGE DETECTION")
    print("=" * 70)

    device = torch.device("cpu")
    checkpoint_file = Path(checkpoint_path)

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}. Train the model first.")

    # 1. Load Model
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model = UNet(in_channels=3, out_channels=1, init_features=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 2. Load T1 and T2 Images
    print(f"Time 1 (Before) Image : {t1_image_path}")
    print(f"Time 2 (After) Image  : {t2_image_path}")
    print(f"Classification Threshold: {threshold}")

    t1_tensor, t1_img_np = load_and_preprocess_image(t1_image_path)
    t2_tensor, t2_img_np = load_and_preprocess_image(t2_image_path)

    # 3. Predict Forest Cover Masks
    t1_mask, t1_probs = predict_forest_mask(model, t1_tensor, device, threshold=threshold)
    t2_mask, t2_probs = predict_forest_mask(model, t2_tensor, device, threshold=threshold)

    # 4. Categorize Pixel Level Temporal Changes
    # Deforestation (Forest Loss): T1 Forest (1) AND T2 Non-Forest (0)
    forest_loss_mask = (t1_mask == 1) & (t2_mask == 0)

    # Forest Gain (Regrowth): T1 Non-Forest (0) AND T2 Forest (1)
    forest_gain_mask = (t1_mask == 0) & (t2_mask == 1)

    # Stable Forest: T1 Forest (1) AND T2 Forest (1)
    stable_forest_mask = (t1_mask == 1) & (t2_mask == 1)

    # Stable Non-Forest: T1 Non-Forest (0) AND T2 Non-Forest (0)
    stable_nonforest_mask = (t1_mask == 0) & (t2_mask == 0)

    # 5. Compute Quantitative Metrics (Pixel Counts & Percentages)
    total_pixels = t1_mask.size
    t1_forest_pixels = int(np.sum(t1_mask == 1))
    t2_forest_pixels = int(np.sum(t2_mask == 1))

    loss_pixels = int(np.sum(forest_loss_mask))
    gain_pixels = int(np.sum(forest_gain_mask))
    stable_forest_pixels = int(np.sum(stable_forest_mask))
    stable_nonforest_pixels = int(np.sum(stable_nonforest_mask))

    t1_forest_pct = (t1_forest_pixels / total_pixels) * 100.0
    t2_forest_pct = (t2_forest_pixels / total_pixels) * 100.0

    # Percentage relative to baseline T1 forest cover
    loss_pct_of_t1_forest = (loss_pixels / t1_forest_pixels * 100.0) if t1_forest_pixels > 0 else 0.0
    gain_pct_of_t1_forest = (gain_pixels / t1_forest_pixels * 100.0) if t1_forest_pixels > 0 else 0.0
    net_forest_change_pct = t2_forest_pct - t1_forest_pct

    stats = {
        "t1_forest_pixels": t1_forest_pixels,
        "t1_forest_pct": t1_forest_pct,
        "t2_forest_pixels": t2_forest_pixels,
        "t2_forest_pct": t2_forest_pct,
        "loss_pixels": loss_pixels,
        "loss_pct_of_t1_forest": loss_pct_of_t1_forest,
        "gain_pixels": gain_pixels,
        "gain_pct_of_t1_forest": gain_pct_of_t1_forest,
        "stable_forest_pixels": stable_forest_pixels,
        "stable_nonforest_pixels": stable_nonforest_pixels,
        "net_forest_change_pct": net_forest_change_pct,
        "total_pixels": total_pixels
    }

    print("\n--- TEMPORAL CHANGE ANALYSIS RESULTS ---")
    print(f"Total Image Pixels            : {total_pixels:,}")
    print(f"Time 1 Forest Cover           : {t1_forest_pixels:,} px ({t1_forest_pct:.2f}%)")
    print(f"Time 2 Forest Cover           : {t2_forest_pixels:,} px ({t2_forest_pct:.2f}%)")
    print(f"Deforestation (Forest Loss)   : {loss_pixels:,} px ({loss_pct_of_t1_forest:.2f}% of T1 forest)")
    print(f"Forest Gain / Regrowth        : {gain_pixels:,} px ({gain_pct_of_t1_forest:.2f}% of T1 forest)")
    print(f"Stable Forest Cover           : {stable_forest_pixels:,} px")
    print(f"Stable Non-Forest Area        : {stable_nonforest_pixels:,} px")
    print(f"Net Forest Change             : {net_forest_change_pct:+.2f}% absolute change")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 6. Save Text Summary Report
    report_path = out_path / "deforestation_change_report.txt"
    with open(report_path, "w") as f:
        f.write("========================================================\n")
        f.write("TEMPORAL DEFORESTATION & CHANGE DETECTION REPORT\n")
        f.write("========================================================\n\n")
        f.write(f"T1 Image (Before) : {t1_image_path}\n")
        f.write(f"T2 Image (After)  : {t2_image_path}\n")
        f.write(f"Model Checkpoint  : {checkpoint_path}\n")
        f.write(f"Threshold         : {threshold}\n\n")
        f.write("QUANTITATIVE CHANGE METRICS:\n")
        f.write(f"- Total Image Pixels        : {total_pixels:,}\n")
        f.write(f"- T1 Forest Cover           : {t1_forest_pixels:,} px ({t1_forest_pct:.2f}%)\n")
        f.write(f"- T2 Forest Cover           : {t2_forest_pixels:,} px ({t2_forest_pct:.2f}%)\n")
        f.write(f"- Forest Loss (Deforested)  : {loss_pixels:,} px ({loss_pct_of_t1_forest:.2f}% of T1 forest)\n")
        f.write(f"- Forest Gain (Regrowth)   : {gain_pixels:,} px ({gain_pct_of_t1_forest:.2f}% of T1 forest)\n")
        f.write(f"- Stable Forest             : {stable_forest_pixels:,} px\n")
        f.write(f"- Stable Non-Forest         : {stable_nonforest_pixels:,} px\n")
        f.write(f"- Net Change in Forest      : {net_forest_change_pct:+.2f}%\n")
    print(f"\nSaved change report to: {report_path}")

    # 7. Create 4-Color Deforestation Map
    # Map Colors:
    # Red [0.9, 0.1, 0.1] = Deforestation (Forest Loss)
    # Blue [0.1, 0.4, 0.9] = Forest Gain
    # Green [0.1, 0.8, 0.2] = Stable Forest
    # Dark Gray [0.2, 0.2, 0.2] = Stable Non-Forest
    change_map_rgb = np.zeros((*t1_mask.shape, 3), dtype=np.float32)
    change_map_rgb[stable_nonforest_mask] = [0.2, 0.2, 0.2]  # Dark Gray
    change_map_rgb[stable_forest_mask] = [0.1, 0.8, 0.2]     # Green
    change_map_rgb[forest_gain_mask] = [0.1, 0.4, 0.9]       # Blue
    change_map_rgb[forest_loss_mask] = [0.9, 0.1, 0.1]       # Red

    # 8. Generate Visual Figure
    fig = plt.figure(figsize=(16, 12))

    # T1 Image & Mask
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(t1_img_np)
    ax1.set_title(f"Time 1 (Before)\nForest: {t1_forest_pct:.1f}% ({t1_forest_pixels:,} px)")
    ax1.axis("off")

    # T2 Image & Mask
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(t2_img_np)
    ax2.set_title(f"Time 2 (After)\nForest: {t2_forest_pct:.1f}% ({t2_forest_pixels:,} px)")
    ax2.axis("off")

    # 4-Color Deforestation Map
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(change_map_rgb)
    ax3.set_title("Temporal Deforestation Change Map\n(Red=Loss, Blue=Gain, Green=Stable, Gray=Non-Forest)")
    ax3.axis("off")

    # Bar Chart Summary
    ax4 = fig.add_subplot(2, 2, 4)
    categories = ["Stable Forest", "Forest Loss", "Forest Gain", "Stable Non-Forest"]
    counts = [stable_forest_pixels, loss_pixels, gain_pixels, stable_nonforest_pixels]
    colors = ["#1ab332", "#e61a1a", "#1a66e6", "#404040"]

    bars = ax4.bar(categories, counts, color=colors, edgecolor="black")
    ax4.set_title("Pixel Category Distribution")
    ax4.set_ylabel("Pixel Count")
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = (height / total_pixels) * 100.0
        ax4.text(bar.get_x() + bar.get_width()/2., height + (max(counts)*0.01),
                 f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    plt.suptitle("AI-Based Temporal Deforestation Change Analysis", fontsize=16, y=0.98)
    plt.tight_layout()
    map_path = out_path / "deforestation_change_map.png"
    plt.savefig(map_path, dpi=150)
    plt.close()

    print(f"Saved temporal change visual map to: {map_path}")
    print("=" * 70)

    return stats


def run_demonstration():
    """
    Demonstrates temporal deforestation detection using sample images from dataset.
    Note: Standard dataset split images represent separate spatial samples.
    This demonstration illustrates how the temporal detection pipeline processes T1 and T2 images.
    """
    test_dir = Path("data/Amazon Forest Dataset/Test")
    test_images = sorted(list(test_dir.glob("*.tiff"))) + sorted(list(test_dir.glob("*.tif")))

    if len(test_images) < 2:
        val_dir = Path("data/Amazon Forest Dataset/Validation/images")
        test_images = sorted(list(val_dir.glob("*.tiff"))) + sorted(list(val_dir.glob("*.tif")))

    t1_file = test_images[0]
    t2_file = test_images[1]

    print("\n[DEMONSTRATION MODE]")
    print("Executing change detection pipeline on sample satellite images:")
    detect_deforestation(t1_file, t2_file)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        t1_path = sys.argv[1]
        t2_path = sys.argv[2]
        detect_deforestation(t1_path, t2_path)
    else:
        run_demonstration()
