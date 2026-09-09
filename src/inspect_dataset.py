from pathlib import Path
from PIL import Image
import numpy as np

# Dataset location
dataset = Path("data/Amazon Forest Dataset")

# Find TIFF files
images = list(dataset.rglob("*.tif")) + list(dataset.rglob("*.tiff"))

print("=" * 50)
print("DATASET INSPECTION")
print("=" * 50)

print(f"\nTotal TIFF files found: {len(images)}")

# Show first 10 files
for i, image_path in enumerate(images[:10]):
    try:
        image = Image.open(image_path)
        array = np.array(image)

        print(f"\n{i + 1}. {image_path}")
        print(f"   Size       : {image.size}")
        print(f"   Mode       : {image.mode}")
        print(f"   Shape      : {array.shape}")
        print(f"   Data type  : {array.dtype}")
        print(f"   Min value  : {array.min()}")
        print(f"   Max value  : {array.max()}")

    except Exception as e:
        print(f"\nCould not read {image_path}")
        print(f"Error: {e}")