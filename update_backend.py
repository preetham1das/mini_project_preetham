import re
from pathlib import Path

with open("backend/main.py", "r", encoding="utf-8") as f:
    code = f.read()

new_imports = """
from fastapi import FastAPI, File, UploadFile, Form
"""
code = code.replace("from fastapi import FastAPI, File, UploadFile", new_imports.strip())

samples_endpoint = """
@app.get("/api/samples")
def get_samples():
    base_dir = Path(__file__).resolve().parent.parent
    amazon_dir = base_dir / "data" / "Amazon Forest Dataset"
    stage1_samples = []
    
    val_dir = amazon_dir / "Validation" / "images"
    if val_dir.exists():
        val_images = sorted(list(val_dir.glob("*.tiff")) + list(val_dir.glob("*.tif")))
        for f in val_images[:10]:
            stage1_samples.append({
                "id": f.name,
                "name": f.name,
                "split": "Validation",
                "path": str(f.resolve())
            })

    test_dir = amazon_dir / "Test"
    if test_dir.exists():
        test_images = sorted(list(test_dir.glob("*.tiff")) + list(test_dir.glob("*.tif")))
        for f in test_images[:10]:
            stage1_samples.append({
                "id": f.name,
                "name": f"Test / {f.name}",
                "split": "Test",
                "path": str(f.resolve())
            })

    temporal_root = base_dir / "data" / "Forest-Change" / "extracted" / "Forest-Change-dataset" / "images"
    stage2_samples = []
    
    test_a_dir = temporal_root / "test" / "A"
    if test_a_dir.exists():
        test_pairs = sorted([f.stem for f in test_a_dir.glob("*.png")])
        for p in test_pairs[:12]:
            stage2_samples.append({
                "id": p,
                "name": f"Test Pair: {p}",
                "split": "test",
                "path_a": str((test_a_dir / f"{p}.png").resolve()),
                "path_b": str((temporal_root / "test" / "B" / f"{p}.png").resolve())
            })
            
    val_a_dir = temporal_root / "val" / "A"
    if val_a_dir.exists():
        val_pairs = sorted([f.stem for f in val_a_dir.glob("*.png")])
        for p in val_pairs[:6]:
            stage2_samples.append({
                "id": p,
                "name": f"Val Pair: {p}",
                "split": "val",
                "path_a": str((val_a_dir / f"{p}.png").resolve()),
                "path_b": str((temporal_root / "val" / "B" / f"{p}.png").resolve())
            })

    return {
        "stage1_samples": stage1_samples,
        "stage2_samples": stage2_samples
    }

@app.post("/predict")
"""
code = code.replace("@app.post(\"/predict\")", samples_endpoint.strip())

predict_sig = """
@app.post("/predict")
async def predict(
    file: UploadFile = File(None),
    sample_path: str = Form(None),
    threshold: float = Form(0.5)
):
    if sample_path:
        try:
            image = Image.open(sample_path).convert("RGB")
            filename = Path(sample_path).name
        except Exception:
            return {"success": False, "error": "Invalid sample path."}
    elif file:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            filename = file.filename
        except Exception:
            return {"success": False, "error": "Invalid image file."}
    else:
        return {"success": False, "error": "No file or sample_path provided."}
"""

predict_temporal_sig = """
@app.post("/predict-temporal")
async def predict_temporal(
    before_image: UploadFile = File(None),
    after_image: UploadFile = File(None),
    sample_path_a: str = Form(None),
    sample_path_b: str = Form(None),
    threshold: float = Form(0.5)
):
    if sample_path_a and sample_path_b:
        try:
            img_a = Image.open(sample_path_a).convert("RGB")
            img_b = Image.open(sample_path_b).convert("RGB")
        except Exception:
            return {"success": False, "error": "Invalid sample paths."}
    elif before_image and after_image:
        try:
            t1_contents = await before_image.read()
            img_a = Image.open(io.BytesIO(t1_contents)).convert("RGB")
            t2_contents = await after_image.read()
            img_b = Image.open(io.BytesIO(t2_contents)).convert("RGB")
        except Exception:
            return {"success": False, "error": "Invalid image file(s)."}
    else:
        return {"success": False, "error": "Provide either uploaded files or sample paths."}
"""

code = re.sub(
    r'@app.post\("/predict"\)\s*async def predict\(\s*file: UploadFile = File\(\.\.\.\)\s*\):\s*# Read uploaded image\s*contents = await file\.read\(\)\s*try:\s*image = Image\.open\(\s*io\.BytesIO\(contents\)\s*\)\.convert\("RGB"\)\s*except Exception:\s*return \{\s*"success": False,\s*"error": "Invalid image file\."\s*\}',
    predict_sig.strip(),
    code
)

code = re.sub(
    r'@app.post\("/predict-temporal"\)\s*async def predict_temporal\(\s*before_image: UploadFile = File\(\.\.\.\),\s*after_image: UploadFile = File\(\.\.\.\)\s*\):\s*try:\s*t1_contents = await before_image\.read\(\)\s*img_a = Image\.open\(io\.BytesIO\(t1_contents\)\)\.convert\("RGB"\)\s*t2_contents = await after_image\.read\(\)\s*img_b = Image\.open\(io\.BytesIO\(t2_contents\)\)\.convert\("RGB"\)\s*except Exception:\s*return \{\s*"success": False,\s*"error": "Invalid image file\(s\)\."\s*\}',
    predict_temporal_sig.strip(),
    code
)

code = code.replace("probabilities > THRESHOLD", "probabilities > threshold")
code = code.replace("\"threshold\": THRESHOLD,", "\"threshold\": threshold,")
code = code.replace("\"filename\": file.filename,", "\"filename\": filename,")

with open("backend/main.py", "w", encoding="utf-8") as f:
    f.write(code)
