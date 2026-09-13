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
    from src.model import CHECKPOINT_PATH, UNet
    from src.dataset import get_dataloaders
    from src.metrics import THRESHOLD
except ModuleNotFoundError:
    from model import CHECKPOINT_PATH, UNet
    from dataset import get_dataloaders
    from metrics import THRESHOLD


def run_inference(
    checkpoint_path=CHECKPOINT_PATH,
    dataset_root="data/Amazon Forest Dataset",
    output_dir="outputs/predictions",
    threshold=THRESHOLD
):
    print("=" * 60)
    print("UNLABELED TEST SET INFERENCE & FOREST COVER PREDICTION")
    print("=" * 60)

    device = torch.device("cpu")
    checkpoint_file = Path(checkpoint_path)

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}. Train the model first.")

    # 1. Load Model
    print(f"Loading model checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)

    model = UNet(in_channels=3, out_channels=1, init_features=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Classification Threshold: {threshold}")

    # 2. Load Test Loader
    _, _, test_loader = get_dataloaders(dataset_root=dataset_root, batch_size=1)
    num_samples = len(test_loader.dataset)
    print(f"Processing {num_samples} unlabeled test images...")

    out_path = Path(output_dir)
    masks_path = out_path / "masks"
    overlays_path = out_path / "overlays"

    masks_path.mkdir(parents=True, exist_ok=True)
    overlays_path.mkdir(parents=True, exist_ok=True)

    test_summaries = []

    with torch.no_grad():
        for images, filenames in test_loader:
            filename = filenames[0] if isinstance(filenames, (list, tuple)) else filenames
            stem = Path(filename).stem

            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            img_np = images[0].permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)

            pred_mask = preds[0, 0].cpu().numpy().astype(np.uint8)
            forest_pixels = int(np.sum(pred_mask == 1))
            total_pixels = int(pred_mask.size)
            forest_percentage = (forest_pixels / total_pixels) * 100.0

            test_summaries.append({
                "filename": filename,
                "stem": stem,
                "forest_pixels": forest_pixels,
                "total_pixels": total_pixels,
                "forest_percentage": forest_percentage,
                "img_np": img_np,
                "pred_mask": pred_mask
            })

            # Save binary mask (0 = Non-forest, 255 = Forest)
            mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8))
            mask_pil.save(masks_path / f"{stem}_predicted_mask.png")

            # Create visual overlay
            overlay = img_np.copy()
            # Highlight forest pixels with semi-transparent green tint
            green_tint = np.array([0.0, 0.8, 0.2], dtype=np.float32)
            alpha = 0.4
            forest_indices = (pred_mask == 1)
            overlay[forest_indices] = (1 - alpha) * overlay[forest_indices] + alpha * green_tint

            # Save visual comparison
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_np)
            axes[0].set_title(f"Test Image: {filename}")
            axes[0].axis("off")

            axes[1].imshow(pred_mask, cmap="Greens", vmin=0, vmax=1)
            axes[1].set_title(f"Predicted Forest Mask\n(Forest: {forest_percentage:.1f}%)")
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title(f"Forest Overlay Map")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(overlays_path / f"{stem}_overlay.png", dpi=150)
            plt.close()

            print(f"Processed: {filename} -> Forest Cover: {forest_percentage:.2f}% ({forest_pixels:,} pixels)")

    # 3. Create Grid Summary Figure
    num_grid = min(9, len(test_summaries))
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(num_grid):
        item = test_summaries[i]
        overlay = item["img_np"].copy()
        forest_indices = (item["pred_mask"] == 1)
        green_tint = np.array([0.0, 0.8, 0.2], dtype=np.float32)
        overlay[forest_indices] = 0.6 * overlay[forest_indices] + 0.4 * green_tint

        axes[i].imshow(overlay)
        axes[i].set_title(f"{item['stem']}\nForest Cover: {item['forest_percentage']:.1f}%")
        axes[i].axis("off")

    plt.suptitle("Test Dataset Forest Segmentation Overlay Summary", fontsize=16, y=0.98)
    plt.tight_layout()
    grid_path = out_path.parent / "test_predictions_summary.png"
    plt.savefig(grid_path, dpi=150)
    plt.close()

    print("\n--- INFERENCE SUMMARY ---")
    print(f"Saved binary prediction masks to : {masks_path}")
    print(f"Saved overlay visual maps to     : {overlays_path}")
    print(f"Saved test summary grid plot to  : {grid_path}")
    print("=" * 60)

    return test_summaries


if __name__ == "__main__":
    run_inference()
