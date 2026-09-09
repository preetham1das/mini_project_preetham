from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Dataset paths
dataset = Path("data/Amazon Forest Dataset")

image_dir = dataset / "Training" / "images"
mask_dir = dataset / "Training" / "masks"

# Pick the first image
image_path = sorted(image_dir.glob("*.tiff"))[0]

# Corresponding mask has the same name, but .png extension
mask_path = mask_dir / (image_path.stem + ".png")

print("=" * 60)
print("SAMPLE IMAGE + MASK")
print("=" * 60)

print("\nImage:", image_path)
print("Mask :", mask_path)

# Load image and mask
image = Image.open(image_path)
mask = Image.open(mask_path)

image_array = np.array(image)
mask_array = np.array(mask)

print("\nImage information:")
print("  Size  :", image.size)
print("  Mode  :", image.mode)
print("  Shape :", image_array.shape)

print("\nMask information:")
print("  Size  :", mask.size)
print("  Mode  :", mask.mode)
print("  Shape :", mask_array.shape)
print("  Values:", np.unique(mask_array))

# Display
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Satellite Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.tight_layout()

# Save result
output_path = Path("outputs/sample_pair.png")
plt.savefig(output_path, dpi=150)

print("\nSaved visualization to:")
print(output_path)

plt.show()