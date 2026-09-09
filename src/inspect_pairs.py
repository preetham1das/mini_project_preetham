from pathlib import Path

dataset = Path("data/Amazon Forest Dataset")

for split in ["Training", "Validation"]:
    print("\n" + "=" * 60)
    print(split.upper())
    print("=" * 60)

    image_dir = dataset / split / "images"
    mask_dir = dataset / split / "masks"

    images = sorted(list(image_dir.glob("*")))
    masks = sorted(list(mask_dir.glob("*")))

    print(f"Images : {len(images)}")
    print(f"Masks  : {len(masks)}")

    print("\nFirst 5 image files:")
    for f in images[:5]:
        print(" ", f.name)

    print("\nFirst 5 mask files:")
    for f in masks[:5]:
        print(" ", f.name)