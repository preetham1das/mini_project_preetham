from pathlib import Path

dataset = Path("data/Amazon Forest Dataset")

print("=" * 60)
print("DATASET STRUCTURE")
print("=" * 60)

for folder in dataset.iterdir():
    if folder.is_dir():
        files = list(folder.rglob("*"))

        file_count = sum(1 for f in files if f.is_file())

        print(f"\n📁 {folder.name}")
        print(f"   Files: {file_count}")

        # Show first 5 files
        shown = 0
        for f in files:
            if f.is_file():
                print(f"   - {f.relative_to(dataset)}")
                shown += 1
                if shown == 5:
                    break