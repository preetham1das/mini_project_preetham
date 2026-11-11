import base64
import io
import logging
from typing import Any

import numpy as np
import torch
from PIL import Image
from pyproj import Geod


logger = logging.getLogger(__name__)
_WGS84_GEOD = Geod(ellps="WGS84")
BENCHMARK_METADATA = {
    "forest_cover_unet": {
        "validation_iou": 0.8385551377629015,
        "validation_dice": 0.9121892735653608,
        "validation_split": "Amazon Forest Dataset validation",
    },
    "temporal_unet": {
        "validation_iou": 0.5042657685060166,
        "validation_dice": 0.6704477081956541,
        "validation_dice_derived_from_iou": True,
        "validation_epoch": 21,
        "validation_split": "Forest-Change validation",
    },
}


def parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    try:
        west, south, east, north = (float(value.strip()) for value in bbox.split(","))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("bbox must contain west,south,east,north coordinates.") from exc

    if not all(np.isfinite(value) for value in (west, south, east, north)):
        raise ValueError("bbox coordinates must be finite numbers.")
    if not (-180 <= west < east <= 180):
        raise ValueError("bbox longitudes must satisfy -180 <= west < east <= 180.")
    if not (-90 <= south < north <= 90):
        raise ValueError("bbox latitudes must satisfy -90 <= south < north <= 90.")

    return west, south, east, north


def geodesic_area_hectares(bbox: tuple[float, float, float, float]) -> float:
    west, south, east, north = bbox
    area_square_meters, _ = _WGS84_GEOD.polygon_area_perimeter(
        [west, east, east, west],
        [south, south, north, north],
    )
    area_hectares = abs(area_square_meters) / 10_000.0
    if not np.isfinite(area_hectares) or area_hectares <= 0:
        raise ValueError("bbox must enclose a non-zero geographic area.")
    return float(area_hectares)


def predict_directional_deforestation(
    forest_model: torch.nn.Module,
    temporal_model: torch.nn.Module,
    image_t1: torch.Tensor,
    image_t2: torch.Tensor,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    if image_t1.shape != image_t2.shape:
        raise ValueError("T1 and T2 tensors must have identical shapes.")
    if image_t1.ndim != 4 or image_t1.shape[1] != 3:
        raise ValueError("T1 and T2 tensors must have shape (batch, 3, height, width).")
    if image_t1.shape[0] != 1:
        raise ValueError("T1 and T2 tensors must contain exactly one image pair.")
    if image_t1.shape[2] <= 0 or image_t1.shape[3] <= 0:
        raise ValueError("T1 and T2 tensors must have non-empty spatial dimensions.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    image_t1 = image_t1.to(dtype=torch.float32)
    image_t2 = image_t2.to(dtype=torch.float32)
    if not torch.isfinite(image_t1).all() or not torch.isfinite(image_t2).all():
        raise ValueError("T1 and T2 tensors must contain only finite values.")
    if image_t1.min() < 0 or image_t1.max() > 1 or image_t2.min() < 0 or image_t2.max() > 1:
        raise ValueError("T1 and T2 tensors must be normalized to [0, 1].")

    forest_model.eval()
    temporal_model.eval()
    with torch.no_grad():
        forest_t1_probs = torch.sigmoid(forest_model(image_t1.to(device)))
        forest_t2_probs = torch.sigmoid(forest_model(image_t2.to(device)))
        expected_shape = (1, 1, image_t1.shape[2], image_t1.shape[3])
        if forest_t1_probs.shape != expected_shape or forest_t2_probs.shape != expected_shape:
            raise RuntimeError("Forest model output must have shape (1, 1, height, width).")
        if not torch.isfinite(forest_t1_probs).all() or not torch.isfinite(forest_t2_probs).all():
            raise RuntimeError("Forest model produced non-finite probability values.")
        mask_t1 = forest_t1_probs[0, 0] >= 0.5
        mask_t2 = forest_t2_probs[0, 0] >= 0.5
        logical_loss_mask = mask_t1 & ~mask_t2

        img_T1 = image_t1.detach().cpu().numpy().transpose(0, 2, 3, 1)
        img_T2 = image_t2.detach().cpu().numpy().transpose(0, 2, 3, 1)
        input_9ch = np.concatenate(
            [img_T1, img_T2, np.abs(img_T1 - img_T2)], axis=-1
        ).astype(np.float32, copy=False)
        temporal_input = torch.from_numpy(
            np.ascontiguousarray(input_9ch.transpose(0, 3, 1, 2))
        ).to(device=device, dtype=torch.float32)
        temporal_probabilities = torch.sigmoid(temporal_model(temporal_input))

    if not torch.isfinite(temporal_probabilities).all():
        raise RuntimeError("Temporal model produced non-finite probability values.")
    if temporal_probabilities.shape != expected_shape:
        raise RuntimeError("Temporal model output must have shape (1, 1, height, width).")

    probability_min = float(temporal_probabilities.min().item())
    probability_max = float(temporal_probabilities.max().item())
    probability_mean = float(temporal_probabilities.mean().item())
    logger.info(
        "Temporal probability stats: min=%.6f max=%.6f mean=%.6f",
        probability_min,
        probability_max,
        probability_mean,
    )

    temporal_mask = temporal_probabilities[0, 0] >= threshold
    final_mask = temporal_mask & logical_loss_mask
    used_logical_fallback = not bool(temporal_mask.any().item())
    if used_logical_fallback:
        final_mask = logical_loss_mask
        logger.warning(
            "Temporal prediction is empty; falling back to the T1-forest/T2-non-forest gate."
        )

    return {
        "mask_t1": mask_t1.cpu().numpy().astype(np.uint8),
        "mask_t2": mask_t2.cpu().numpy().astype(np.uint8),
        "temporal_mask": temporal_mask.cpu().numpy().astype(np.uint8),
        "final_mask": final_mask.cpu().numpy().astype(np.uint8),
        "logical_gate_pixels": int(logical_loss_mask.sum().item()),
        "temporal_prediction_pixels": int(temporal_mask.sum().item()),
        "temporal_probability_min": probability_min,
        "temporal_probability_max": probability_max,
        "temporal_probability_mean": probability_mean,
        "temporal_threshold_used": threshold,
        "used_logical_fallback": used_logical_fallback,
    }


def rgba_mask_to_data_uri(mask: np.ndarray) -> str:
    binary_mask = np.asarray(mask, dtype=bool)
    rgba = np.zeros((*binary_mask.shape, 4), dtype=np.uint8)
    rgba[binary_mask] = (255, 0, 0, 180)

    buffer = io.BytesIO()
    Image.fromarray(rgba).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
