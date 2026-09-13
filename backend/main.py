from pathlib import Path
import io
import base64

import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model import UNet
from src.temporal_model import TemporalUNet

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device("cpu")
IMAGE_SIZE = 256
THRESHOLD = 0.5

CHECKPOINT_PATH = Path("models/best_unet_model.pth")
TEMPORAL_CHECKPOINT_PATH = Path("models/best_temporal_model.pth")

# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Forest Segmentation API",
    description="AI-based forest cover segmentation using U-Net"
)


# Allow React frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Load models once when server starts
# --------------------------------------------------

print("Loading Stage 1 U-Net model...")

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model = UNet(
    in_channels=3,
    out_channels=1,
    init_features=32
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Stage 1 U-Net model loaded successfully.")


print("Loading Stage 2 Temporal U-Net model...")
if not TEMPORAL_CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {TEMPORAL_CHECKPOINT_PATH}"
    )

temporal_checkpoint = torch.load(
    TEMPORAL_CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

temporal_model = TemporalUNet(
    in_channels=6,
    out_channels=1,
    init_features=32
).to(DEVICE)

temporal_model.load_state_dict(temporal_checkpoint["model_state_dict"])
temporal_model.eval()

print("Stage 2 Temporal U-Net model loaded successfully.")

# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image: Image.Image):

    image = image.convert("RGB")

    # Resize exactly like training
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(image).astype(np.float32) / 255.0

    # H,W,C -> C,H,W
    image_tensor = torch.from_numpy(
        image_array
    ).permute(2, 0, 1)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


# --------------------------------------------------
# Convert mask to base64 PNG
# --------------------------------------------------

def mask_to_base64(mask):

    mask_image = Image.fromarray(
        (mask * 255).astype(np.uint8)
    )

    buffer = io.BytesIO()

    mask_image.save(
        buffer,
        format="PNG"
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# --------------------------------------------------
# Convert image to base64 PNG
# --------------------------------------------------

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# --------------------------------------------------
# API endpoints
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Forest Segmentation API is running"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Read uploaded image
    contents = await file.read()

    try:
        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")
    except Exception:
        return {
            "success": False,
            "error": "Invalid image file."
        }

    # Original dimensions
    original_width, original_height = image.size

    # Preprocess
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(DEVICE)

    # --------------------------------------------------
    # Model prediction
    # --------------------------------------------------

    with torch.no_grad():

        logits = model(image_tensor)

        probabilities = torch.sigmoid(logits)

        prediction = (
            probabilities > THRESHOLD
        ).float()

    # Get mask
    mask = prediction[
        0, 0
    ].cpu().numpy().astype(np.uint8)

    # --------------------------------------------------
    # Forest percentage
    # --------------------------------------------------

    forest_pixels = int(
        np.sum(mask == 1)
    )

    total_pixels = int(
        mask.size
    )

    forest_percentage = (
        forest_pixels / total_pixels
    ) * 100

    # --------------------------------------------------
    # Return result
    # --------------------------------------------------

    return {
        "success": True,
        "filename": file.filename,
        "original_width": original_width,
        "original_height": original_height,
        "forest_percentage": round(
            forest_percentage,
            2
        ),
        "forest_pixels": forest_pixels,
        "total_pixels": total_pixels,
        "threshold": THRESHOLD,
        "mask": mask_to_base64(mask),
        "original_image": image_to_base64(image)
    }


@app.post("/predict-temporal")
async def predict_temporal(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...)
):
    try:
        t1_contents = await before_image.read()
        img_a = Image.open(io.BytesIO(t1_contents)).convert("RGB")

        t2_contents = await after_image.read()
        img_b = Image.open(io.BytesIO(t2_contents)).convert("RGB")
    except Exception:
        return {
            "success": False,
            "error": "Invalid image file(s)."
        }

    # Compatible dimensions check
    if img_a.size != img_b.size:
        return {
            "success": False,
            "error": "Images must have identical dimensions for accurate temporal comparison."
        }

    # Preprocess both exactly as during training
    img_a_tensor = preprocess_image(img_a)
    img_b_tensor = preprocess_image(img_b)

    # Concatenate (1, 6, 256, 256)
    temporal_tensor = torch.cat([img_a_tensor, img_b_tensor], dim=1).to(DEVICE)

    with torch.no_grad():
        logits = temporal_model(temporal_tensor)
        probabilities = torch.sigmoid(logits)
        prediction = (probabilities > THRESHOLD).float()

    mask = prediction[0, 0].cpu().numpy().astype(np.uint8)

    changed_pixels = int(np.sum(mask == 1))
    total_pixels = int(mask.size)
    deforestation_percentage = (changed_pixels / total_pixels) * 100

    return {
        "success": True,
        "before_image": image_to_base64(img_a),
        "after_image": image_to_base64(img_b),
        "deforestation_mask": mask_to_base64(mask),
        "deforestation_percentage": round(deforestation_percentage, 2),
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels
    }
