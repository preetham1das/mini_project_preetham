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


def sentinel_bands_to_reflectance(
    bands: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert B04/B03/B02/B08/SCL band-first data to reflectance and metadata."""
    source = np.asarray(bands)
    if source.ndim != 3 or source.shape[0] < 4:
        raise ValueError("Sentinel imagery must contain B04, B03, B02, and B08 bands.")
    if source.shape[1] <= 0 or source.shape[2] <= 0:
        raise ValueError("Sentinel bands must have non-empty spatial dimensions.")

    reflectance = source[:4].astype(np.float32, copy=True)
    if not np.all(np.isfinite(reflectance)):
        raise ValueError("Sentinel reflectance bands must contain only finite values.")
    if float(np.max(np.abs(reflectance))) > 1.0:
        reflectance /= 10_000.0
    rgb = np.moveaxis(reflectance[:3], 0, -1)
    spectral = {
        "red": reflectance[0],
        "nir": reflectance[3],
    }
    if source.shape[0] >= 5:
        spectral["scl"] = source[4].astype(np.float32, copy=False)
    return rgb, spectral


def _forest_mask_and_vegetation(
    probabilities: np.ndarray,
    rgb: np.ndarray,
    nir: np.ndarray | None = None,
    red_band: np.ndarray | None = None,
    scl: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    probability_map = np.asarray(probabilities, dtype=np.float32)
    rgb_image = np.asarray(rgb, dtype=np.float32)
    if probability_map.ndim != 2 or probability_map.size == 0:
        raise ValueError("Forest probabilities must be a non-empty 2D array.")
    if rgb_image.shape != (*probability_map.shape, 3):
        raise ValueError("Forest probabilities and RGB imagery must have matching 2D spatial dimensions.")
    if not np.all(np.isfinite(probability_map)) or not np.all(np.isfinite(rgb_image)):
        raise ValueError("Forest probabilities and RGB imagery must contain only finite values.")

    probability_cutoff = np.percentile(probability_map, 35)
    if nir is not None:
        nir_band = np.asarray(nir, dtype=np.float32)
        red = rgb_image[..., 0] if red_band is None else np.asarray(red_band, dtype=np.float32)
        if nir_band.shape != probability_map.shape or red.shape != probability_map.shape:
            raise ValueError("NIR and red bands must match the forest probability dimensions.")
        if not np.all(np.isfinite(nir_band)) or not np.all(np.isfinite(red)):
            raise ValueError("NIR and red bands must contain only finite values.")
        vegetation_index = (nir_band - red) / (nir_band + red + 1e-6)
        forest_mask = (probability_map > probability_cutoff) & (vegetation_index > 0.35)
        method = "ndvi"
    else:
        vegetation_index = 2.0 * rgb_image[..., 1] - rgb_image[..., 0] - rgb_image[..., 2]
        vegetation_cutoff = np.percentile(vegetation_index, 40)
        forest_mask = (
            (probability_map > probability_cutoff)
            & (vegetation_index > vegetation_cutoff)
        )
        method = "exg"

    if scl is not None:
        scl_values = np.asarray(scl)
        if scl_values.shape != probability_map.shape:
            raise ValueError("SCL must match the forest probability dimensions.")
        forest_mask &= scl_values == 4
    if valid is not None:
        valid_pixels = np.asarray(valid, dtype=bool)
        if valid_pixels.shape != probability_map.shape:
            raise ValueError("Validity mask must match the forest probability dimensions.")
        forest_mask &= valid_pixels
    return forest_mask.astype(np.uint8), vegetation_index.astype(np.float32), method


def refine_forest_mask(
    probabilities: np.ndarray,
    rgb: np.ndarray,
    nir: np.ndarray | None = None,
    red_band: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the legacy refinement used by the separate live Flask route."""
    probability_map = np.asarray(probabilities, dtype=np.float32)
    rgb_image = np.asarray(rgb, dtype=np.float32)
    if probability_map.ndim != 2 or rgb_image.shape != (*probability_map.shape, 3):
        raise ValueError("Forest probabilities and RGB imagery must have matching 2D spatial dimensions.")
    if not np.all(np.isfinite(probability_map)) or not np.all(np.isfinite(rgb_image)):
        raise ValueError("Forest probabilities and RGB imagery must contain only finite values.")

    if nir is not None:
        nir_band = np.asarray(nir, dtype=np.float32)
        red = rgb_image[..., 0] if red_band is None else np.asarray(red_band, dtype=np.float32)
        if nir_band.shape != probability_map.shape or red.shape != probability_map.shape:
            raise ValueError("NIR and red bands must match the forest probability dimensions.")
        if not np.all(np.isfinite(nir_band)) or not np.all(np.isfinite(red)):
            raise ValueError("NIR and red bands must contain only finite values.")
        ndvi = (nir_band - red) / (nir_band + red + 1e-6)
        return ((probability_map > 0.65) & (ndvi > 0.45)).astype(np.uint8)

    excess_green = 2.0 * rgb_image[..., 1] - rgb_image[..., 0] - rgb_image[..., 2]
    return ((probability_map > 0.75) & (excess_green > 0.0)).astype(np.uint8)


def predict_directional_deforestation(
    forest_model: torch.nn.Module,
    temporal_model: torch.nn.Module,
    image_t1: torch.Tensor,
    image_t2: torch.Tensor,
    device: torch.device,
    threshold: float,
    spectral_t1: dict[str, np.ndarray] | None = None,
    spectral_t2: dict[str, np.ndarray] | None = None,
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
        image_t1_rgb = image_t1[0].detach().cpu().permute(1, 2, 0).numpy()
        image_t2_rgb = image_t2[0].detach().cpu().permute(1, 2, 0).numpy()
        use_ndvi = (
            spectral_t1 is not None
            and spectral_t2 is not None
            and "nir" in spectral_t1
            and "nir" in spectral_t2
        )
        nir_t1 = spectral_t1["nir"] if use_ndvi else None
        nir_t2 = spectral_t2["nir"] if use_ndvi else None
        red_t1 = spectral_t1.get("red") if spectral_t1 is not None else None
        red_t2 = spectral_t2.get("red") if spectral_t2 is not None else None
        scl_t1 = spectral_t1.get("scl") if spectral_t1 is not None else None
        scl_t2 = spectral_t2.get("scl") if spectral_t2 is not None else None
        valid_t1 = spectral_t1.get("valid") if spectral_t1 is not None else None
        valid_t2 = spectral_t2.get("valid") if spectral_t2 is not None else None

        mask_t1_np, vegetation_t1, vegetation_mode_t1 = _forest_mask_and_vegetation(
            forest_t1_probs[0, 0].detach().cpu().numpy(),
            image_t1_rgb,
            nir_t1,
            red_t1 if use_ndvi else None,
            scl_t1,
            valid_t1,
        )
        mask_t2_np, vegetation_t2, vegetation_mode_t2 = _forest_mask_and_vegetation(
            forest_t2_probs[0, 0].detach().cpu().numpy(),
            image_t2_rgb,
            nir_t2,
            red_t2 if use_ndvi else None,
            scl_t2,
            valid_t2,
        )
        mask_t1 = torch.from_numpy(mask_t1_np.astype(bool)).to(device=device)
        mask_t2 = torch.from_numpy(mask_t2_np.astype(bool)).to(device=device)
        pcc_loss_mask = mask_t1 & ~mask_t2
        vegetation_drop_mask = (
            (vegetation_t1 - vegetation_t2) > (0.08 if use_ndvi else 0.05)
        )
        if valid_t1 is not None:
            vegetation_drop_mask &= np.asarray(valid_t1, dtype=bool)
        if valid_t2 is not None:
            vegetation_drop_mask &= np.asarray(valid_t2, dtype=bool)

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

    temporal_probability_map = temporal_probabilities[0, 0]
    temporal_candidates = temporal_probability_map >= threshold
    vegetation_drop = torch.from_numpy(vegetation_drop_mask).to(
        device=device, dtype=torch.bool
    )
    temporal_loss_mask = temporal_candidates & vegetation_drop
    final_mask = temporal_loss_mask | pcc_loss_mask
    forest_t1_percent = 100.0 * float(mask_t1_np.mean())
    forest_t2_percent = 100.0 * float(mask_t2_np.mean())
    final_deforestation_pixels = int(final_mask.sum().item())
    logger.info("Stage 1 T1 forest %%: %.2f", forest_t1_percent)
    logger.info("Stage 1 T2 forest %%: %.2f", forest_t2_percent)
    logger.info("Temporal max prob: %.6f", probability_max)
    logger.info("Final deforestation pixels: %d", final_deforestation_pixels)

    return {
        "mask_t1": mask_t1_np,
        "mask_t2": mask_t2_np,
        "deforestation_transition_mask": pcc_loss_mask.cpu().numpy().astype(np.uint8),
        "pcc_loss_mask": pcc_loss_mask.cpu().numpy().astype(np.uint8),
        "vegetation_drop_mask": vegetation_drop_mask.astype(np.uint8),
        "temporal_mask": temporal_loss_mask.cpu().numpy().astype(np.uint8),
        "final_mask": final_mask.cpu().numpy().astype(np.uint8),
        "logical_gate_pixels": int(pcc_loss_mask.sum().item()),
        "temporal_prediction_pixels": int(temporal_candidates.sum().item()),
        "temporal_loss_pixels": int(temporal_loss_mask.sum().item()),
        "stage1_t1_forest_pct": forest_t1_percent,
        "stage1_t2_forest_pct": forest_t2_percent,
        "vegetation_index_method": (
            "ndvi"
            if vegetation_mode_t1 == "ndvi" and vegetation_mode_t2 == "ndvi"
            else "exg"
        ),
        "temporal_probability_min": probability_min,
        "temporal_probability_max": probability_max,
        "temporal_probability_mean": probability_mean,
        "temporal_threshold_used": threshold,
    }


def forest_change_metrics(
    mask_t1: np.ndarray,
    mask_t2: np.ndarray,
    total_area_hectares: float | None = None,
    valid_mask: np.ndarray | None = None,
    deforestation_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    source_t1 = np.asarray(mask_t1)
    source_t2 = np.asarray(mask_t2)
    if source_t1.shape != source_t2.shape or source_t1.ndim != 2:
        raise ValueError("T1 and T2 forest masks must have identical 2D shapes.")
    if np.any((source_t1 != 0) & (source_t1 != 1)) or np.any((source_t2 != 0) & (source_t2 != 1)):
        raise ValueError("T1 and T2 forest masks must contain only binary 0/1 values.")
    forest_t1 = np.array(source_t1, dtype=bool, copy=True)
    forest_t2 = np.array(source_t2, dtype=bool, copy=True)
    if total_area_hectares is not None and (
        not np.isfinite(total_area_hectares) or total_area_hectares <= 0
    ):
        raise ValueError("total_area_hectares must be a finite positive number.")

    valid = np.ones(forest_t1.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != forest_t1.shape:
        raise ValueError("valid_mask must match the forest mask dimensions.")

    forest_t1 &= valid
    forest_t2 &= valid
    logical_deforestation = forest_t1 & ~forest_t2
    if deforestation_mask is None:
        deforestation = logical_deforestation
    else:
        source_deforestation = np.asarray(deforestation_mask)
        if source_deforestation.shape != forest_t1.shape:
            raise ValueError("deforestation_mask must match the forest mask dimensions.")
        if np.any((source_deforestation != 0) & (source_deforestation != 1)):
            raise ValueError("deforestation_mask must contain only binary 0/1 values.")
        deforestation = np.array(source_deforestation, dtype=bool, copy=True)
        deforestation &= valid
    forest_pixels_t1 = int(forest_t1.sum())
    forest_pixels_t2 = int(forest_t2.sum())
    deforested_pixels = int(deforestation.sum())
    signal = "DEFORESTATION" if deforested_pixels else "NO_DEFORESTATION"

    pixel_area_hectares = (
        total_area_hectares / int(valid.sum())
        if total_area_hectares is not None and valid.any()
        else None
    )
    area_unit = "ha" if pixel_area_hectares is not None else "pixels"

    def area_value(pixel_count: int) -> float | int:
        if pixel_area_hectares is None:
            return pixel_count
        return round(pixel_count * pixel_area_hectares, 4)

    return {
        "signal": signal,
        "status": "Deforestation Detected" if deforested_pixels else "No Deforestation Detected",
        "area_unit": area_unit,
        "forest_area_t1": area_value(forest_pixels_t1),
        "forest_area_t2": area_value(forest_pixels_t2),
        "total_forest_area_t1": area_value(forest_pixels_t1),
        "total_forest_area_t2": area_value(forest_pixels_t2),
        "deforested_area": area_value(deforested_pixels),
        "forest_area_t1_hectares": area_value(forest_pixels_t1) if pixel_area_hectares is not None else None,
        "forest_area_t2_hectares": area_value(forest_pixels_t2) if pixel_area_hectares is not None else None,
        "deforested_area_hectares": area_value(deforested_pixels) if pixel_area_hectares is not None else None,
        "forest_pixels_t1": forest_pixels_t1,
        "forest_pixels_t2": forest_pixels_t2,
        "deforested_pixels": deforested_pixels,
        "deforestation_transition_mask": deforestation.astype(np.uint8),
        "active_deforestation_mask": deforestation.astype(np.uint8),
    }


def rgba_mask_to_data_uri(mask: np.ndarray) -> str:
    binary_mask = np.asarray(mask, dtype=bool)
    rgba = np.zeros((*binary_mask.shape, 4), dtype=np.uint8)
    rgba[binary_mask] = (255, 0, 0, 190)

    buffer = io.BytesIO()
    Image.fromarray(rgba).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
