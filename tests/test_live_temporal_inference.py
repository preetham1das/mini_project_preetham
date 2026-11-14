"""Mocked contract tests for the direct live Sentinel-2 Temporal U-Net path."""
import base64
import io
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

import app as forest_app
from temporal_model import TemporalUNet


def make_analysis_tiff(all_invalid=False, shift=0.0, nir=0.8, include_nir=True, rgb=(0.2, 0.15, 0.1)):
    height = width = 480
    transform = from_bounds(-62.0 + shift, -3.0, -61.9 + shift, -2.9, width, height)
    profile = {
        "driver": "GTiff", "width": width, "height": height,
        "count": 5 if include_nir else 4, "dtype": "float32", "crs": "EPSG:4326",
        "transform": transform, "nodata": -9999.0,
    }
    data = np.zeros((5 if include_nir else 4, height, width), dtype=np.float32)
    data[:3] = np.asarray(rgb, dtype=np.float32)[:, None, None]
    validity_index = 4 if include_nir else 3
    if include_nir:
        data[3] = nir
    data[validity_index] = 1.0
    data[validity_index, :, -1] = 0.0
    if all_invalid:
        data[validity_index] = 0.0
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(data)
        return mem.read()


def make_rgb_data_url(rgb):
    image = Image.new("RGB", (32, 32), rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code
        self.ok = 200 <= status_code < 300


class ConstantLogitModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.seen_shape = None

    def forward(self, images):
        self.seen_shape = tuple(images.shape)
        return torch.full((images.shape[0], 1, images.shape[2], images.shape[3]), self.value, device=images.device)


class TemporalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.old_temporal = forest_app.temporal_model
        self.old_stage1 = forest_app.stage1_model
        forest_app.app.config["TESTING"] = True
        self.client = forest_app.app.test_client()

    def tearDown(self):
        forest_app.temporal_model = self.old_temporal
        forest_app.stage1_model = self.old_stage1

    def test_legacy_checkpoint_decoder_names_load_strictly_without_tensor_changes(self):
        source = TemporalUNet(in_channels=9, init_features=2).state_dict()
        legacy = {}
        for name, value in source.items():
            for index in range(1, 5):
                if name.startswith(f"conv_up{index}."):
                    name = f"cu{index}." + name[len(f"conv_up{index}."):]
                    break
            legacy[name] = value.clone()
        mapped = forest_app.remap_temporal_checkpoint_keys(legacy)
        model = TemporalUNet(in_channels=9, init_features=2)
        model.load_state_dict(mapped, strict=True)
        for name, value in source.items():
            torch.testing.assert_close(model.state_dict()[name], value)

    def test_load_models_accepts_the_existing_temporal_checkpoint_format(self):
        previous_temporal = forest_app.temporal_model
        previous_stage1 = forest_app.stage1_model
        previous_iou = forest_app.temporal_model_validation_iou
        model = TemporalUNet(in_channels=9, init_features=2)
        legacy = {}
        for name, value in model.state_dict().items():
            for index in range(1, 5):
                if name.startswith(f"conv_up{index}."):
                    name = f"cu{index}." + name[len(f"conv_up{index}."):]
                    break
            legacy[name] = value
        try:
            with tempfile.TemporaryDirectory() as directory:
                checkpoint_path = Path(directory) / "temporal.pth"
                torch.save({
                    "model_state_dict": legacy,
                    "best_val_iou": 0.5,
                    "config": {"in_channels": 9, "diff_channels": True,
                               "init_features": 2, "bottleneck_dropout": 0.0},
                }, checkpoint_path)
                with patch.object(forest_app, "TEMPORAL_MODEL_CHECKPOINT", checkpoint_path), \
                     patch.object(forest_app, "CHECKPOINT_PATH", Path(directory) / "missing-stage1.pth"):
                    forest_app.load_models()
            self.assertIsInstance(forest_app.temporal_model, TemporalUNet)
            self.assertEqual(forest_app.temporal_model_validation_iou, 0.5)
        finally:
            forest_app.temporal_model = previous_temporal
            forest_app.stage1_model = previous_stage1
            forest_app.temporal_model_validation_iou = previous_iou

    def test_temporal_input_channel_order_and_difference(self):
        t1 = torch.zeros(1, 3, 4, 5)
        t2 = torch.full((1, 3, 4, 5), 0.25)
        value = forest_app.build_temporal_input(t1, t2)
        self.assertEqual(tuple(value.shape), (1, 9, 4, 5))
        self.assertTrue(torch.all(value[:, :3] == 0))
        self.assertTrue(torch.all(value[:, 3:6] == 0.25))
        self.assertTrue(torch.all(value[:, 6:] == 0.25))

    def test_compare_fetches_selected_scenes_and_returns_both_masks(self):
        temporal = ConstantLogitModel(10.0)
        stage1 = ConstantLogitModel(10.0)
        forest_app.temporal_model = temporal
        forest_app.stage1_model = stage1
        tiff_bytes = make_analysis_tiff()
        acquisitions = []

        def fake_post(url, **kwargs):
            acquisitions.append((url, kwargs["json"]))
            return FakeResponse(tiff_bytes)

        payload = {
            "acquisition_before": {"id": "S2_T1", "datetime": "2023-01-01T10:00:00Z", "collection": "sentinel-2-l2a"},
            "acquisition_after": {"id": "S2_T2", "datetime": "2024-01-01T10:00:00Z", "collection": "sentinel-2-l2a"},
            "bbox": [-62.0, -3.0, -61.9, -2.9],
            "threshold": 0.5,
            "detection_method": "compare",
        }
        with patch.object(forest_app, "get_sentinel_token", return_value=("test-token", None, 200)), \
             patch.object(forest_app.requests, "post", side_effect=fake_post):
            response = self.client.post("/api/sentinel/analyze", json=payload)

        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertEqual(len(acquisitions), 2)
        self.assertTrue(all(url.endswith("/process") for url, _ in acquisitions))
        self.assertTrue(all(spec["output"]["width"] == 480 for _, spec in acquisitions))
        self.assertTrue(all('"SCL"' in spec["evalscript"] for _, spec in acquisitions))
        self.assertTrue(all('"B08"' in spec["evalscript"] for _, spec in acquisitions))
        self.assertEqual(result["forest_mask_refinement"]["method"], "NDVI")
        self.assertEqual(temporal.seen_shape, (1, 9, 256, 256))
        self.assertEqual(stage1.seen_shape, (1, 3, 256, 256))
        self.assertEqual(result["temporal_deforestation_pixels"], 0)
        self.assertEqual(result["baseline_deforestation_pixels"], 0)
        self.assertEqual(result["temporal_deforestation_pct"], 0.0)
        self.assertEqual(result["status"], "No Deforestation Detected")
        self.assertEqual(result["deforestation_hectares"], 0.0)
        self.assertGreaterEqual(result["baseline_deforestation_hectares"], 0.0)
        self.assertEqual(result["detection_method"], "compare")
        self.assertEqual(result["model_used"], "temporal_unet_and_stage1_post_classification")
        self.assertTrue(result["checkpoint_info"]["temporal_unet"]["loaded"])
        self.assertEqual(result["prediction_output"]["size"], [256, 256])
        self.assertEqual(result["prediction_output"]["invalid_pixel_encoding"]["mask_rgb"], [128, 128, 128])
        self.assertTrue(result["validity"]["area_excludes_invalid_pixels"])
        self.assertTrue(any("Temporal U-Net" in warning for warning in result["warnings"]))
        self.assertTrue(any("Stage-1 U-Net" in warning for warning in result["warnings"]))
        self.assertEqual(result["geospatial"]["crs"], "EPSG:4326")
        self.assertEqual(result["geospatial"]["prediction_size"], [256, 256])
        self.assertGreater(result["geospatial"]["prediction_gsd"]["horizontal_m_per_pixel_at_center_latitude"], 0)
        self.assertLess(result["valid_pixel_pct"], 100)
        before_png = base64.b64decode(result["before_image"].split(",", 1)[1])
        with Image.open(io.BytesIO(before_png)) as image:
            self.assertEqual(image.size, (256, 256))
            np.testing.assert_array_equal(np.asarray(image)[100, 100], [128, 96, 64])
        self.assertTrue(result["temporal_deforestation_mask"].startswith("data:image/png;base64,"))
        mask_png = base64.b64decode(result["temporal_deforestation_mask"].split(",", 1)[1])
        with Image.open(io.BytesIO(mask_png)) as image:
            self.assertEqual(image.size, (256, 256))
            self.assertEqual(image.mode, "RGB")
        self.assertIn("class 1 is dataset-labelled deforestation", result["model_note"].lower())
        self.assertIn("source-rendering equivalence", result["model_note"].lower())

    def test_live_sentinel_uses_ndvi_loss_intersected_with_temporal_prediction(self):
        temporal = ConstantLogitModel(10.0)
        stage1 = ConstantLogitModel(10.0)
        forest_app.temporal_model = temporal
        forest_app.stage1_model = stage1
        responses = [
            FakeResponse(make_analysis_tiff(nir=0.8)),
            FakeResponse(make_analysis_tiff(nir=0.2)),
        ]
        payload = {
            "acquisition_before": {"id": "S2_T1", "datetime": "2023-01-01T10:00:00Z", "collection": "sentinel-2-l2a"},
            "acquisition_after": {"id": "S2_T2", "datetime": "2024-01-01T10:00:00Z", "collection": "sentinel-2-l2a"},
            "bbox": [-62.0, -3.0, -61.9, -2.9],
            "threshold": 0.5,
            "detection_method": "compare",
        }
        with patch.object(forest_app, "get_sentinel_token", return_value=("test-token", None, 200)), \
             patch.object(forest_app.requests, "post", side_effect=responses):
            response = self.client.post("/api/sentinel/analyze", json=payload)

        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertEqual(result["forest_mask_refinement"]["method"], "NDVI")
        self.assertEqual(result["signal"], "DEFORESTATION")
        self.assertEqual(result["status"], "Deforestation Detected")
        self.assertEqual(result["temporal_deforestation_pixels"], result["baseline_deforestation_pixels"])
        self.assertGreater(result["deforested_pixels"], 0)
        self.assertGreater(result["deforestation_hectares"], 0.0)

    def test_sentinel_rgb_fallback_uses_excess_green_and_the_temporal_gate(self):
        temporal = ConstantLogitModel(10.0)
        stage1 = ConstantLogitModel(10.0)
        forest_app.temporal_model = temporal
        forest_app.stage1_model = stage1
        responses = [
            FakeResponse(b"", status_code=400),
            FakeResponse(make_analysis_tiff(include_nir=False, rgb=(0.05, 0.5, 0.1))),
            FakeResponse(b"", status_code=400),
            FakeResponse(make_analysis_tiff(include_nir=False, rgb=(0.5, 0.25, 0.1))),
        ]
        payload = {
            "acquisition_before": {"id": "S2_T1", "datetime": "2023-01-01T10:00:00Z", "collection": "sentinel-2-l2a"},
            "acquisition_after": {"id": "S2_T2", "datetime": "2024-01-01T10:00:00Z", "collection": "sentinel-2-l2a"},
            "bbox": [-62.0, -3.0, -61.9, -2.9],
            "threshold": 0.5,
            "detection_method": "compare",
        }
        with patch.object(forest_app, "get_sentinel_token", return_value=("test-token", None, 200)), \
             patch.object(forest_app.requests, "post", side_effect=responses) as process_request:
            response = self.client.post("/api/sentinel/analyze", json=payload)

        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertEqual(result["forest_mask_refinement"]["method"], "Excess Green")
        self.assertFalse(result["forest_mask_refinement"]["nir_available"])
        self.assertEqual(result["status"], "Deforestation Detected")
        self.assertGreater(result["deforested_pixels"], 0)
        self.assertTrue('"B08"' in process_request.call_args_list[0].kwargs["json"]["evalscript"])

    def test_stage1_baseline_remains_available_without_temporal_checkpoint(self):
        stage1 = ConstantLogitModel(-10.0)
        forest_app.temporal_model = None
        forest_app.stage1_model = stage1
        tiff_bytes = make_analysis_tiff()
        payload = {
            "acquisition_before": {"id": "S2_T1", "datetime": "2023-01-01T10:00:00Z", "collection": "sentinel-2-l2a"},
            "acquisition_after": {"id": "S2_T2", "datetime": "2024-01-01T10:00:00Z", "collection": "sentinel-2-l2a"},
            "bbox": [-62.0, -3.0, -61.9, -2.9],
            "detection_method": "baseline",
        }
        with patch.object(forest_app, "get_sentinel_token", return_value=("test-token", None, 200)), \
             patch.object(forest_app.requests, "post", return_value=FakeResponse(tiff_bytes)):
            response = self.client.post("/api/sentinel/analyze", json=payload)
        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertEqual(result["model_name"], "Stage 1 post-classification baseline")
        self.assertIsNone(result["temporal_deforestation_mask"])
        self.assertIsNone(result["temporal_validation_iou"])
        self.assertEqual(stage1.seen_shape, (1, 3, 256, 256))

    def test_live_analysis_requires_l2a_for_cloud_masking(self):
        forest_app.temporal_model = ConstantLogitModel(0.0)
        forest_app.stage1_model = ConstantLogitModel(0.0)
        payload = {
            "acquisition_before": {"id": "T1", "datetime": "2023-01-01T10:00:00Z", "collection": "sentinel-2-l1c"},
            "acquisition_after": {"id": "T2", "datetime": "2024-01-01T10:00:00Z", "collection": "sentinel-2-l1c"},
            "bbox": [-62.0, -3.0, -61.9, -2.9],
        }
        with patch.object(forest_app, "get_sentinel_token") as token_request:
            response = self.client.post("/api/sentinel/analyze", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("requires Sentinel-2 L2A", response.get_json()["error"])
        token_request.assert_not_called()

    def test_legacy_payload_defaults_explicitly_to_temporal_and_reports_metadata(self):
        temporal = ConstantLogitModel(10.0)
        stage1 = ConstantLogitModel(-10.0)
        forest_app.temporal_model = temporal
        forest_app.stage1_model = stage1
        response = self.client.post("/api/sentinel/analyze", json={
            "image_before": make_rgb_data_url((20, 40, 60)),
            "image_after": make_rgb_data_url((40, 60, 80)),
            "date_before": "2023-01-01",
            "date_after": "2024-01-01",
            "bbox": [-62.0, -3.0, -61.9, -2.9],
        })
        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertEqual(result["detection_method"], "temporal")
        self.assertTrue(result["method_selection"]["default_applied"])
        self.assertEqual(result["model_used"], "temporal_unet_with_stage1_loss_gate")
        self.assertTrue(result["checkpoint_info"]["temporal_unet"]["loaded"])
        self.assertTrue(result["checkpoint_info"]["stage1"]["loaded"])
        self.assertEqual(result["prediction_output"]["size"], [256, 256])
        self.assertTrue(result["prediction_output"]["all_pixels_treated_as_valid"])
        self.assertTrue(result["warnings"])
        self.assertEqual(temporal.seen_shape, (1, 9, 256, 256))
        self.assertEqual(stage1.seen_shape, (1, 3, 256, 256))

    def test_partial_acquisition_request_uses_complete_image_pair_fallback(self):
        temporal = ConstantLogitModel(10.0)
        forest_app.stage1_model = ConstantLogitModel(10.0)
        forest_app.temporal_model = temporal
        response = self.client.post("/api/sentinel/analyze", json={
            "acquisition_before": {"id": "T1", "datetime": "2023-01-01T00:00:00Z"},
            "image_before": make_rgb_data_url((20, 40, 60)),
            "image_after": make_rgb_data_url((40, 60, 80)),
            "date_before": "2023-01-01",
            "date_after": "2024-01-01",
            "detection_method": "temporal",
        })
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["model_used"], "temporal_unet_with_stage1_loss_gate")
        self.assertEqual(temporal.seen_shape, (1, 9, 256, 256))

    def test_missing_both_input_pairs_returns_actionable_error(self):
        response = self.client.post("/api/sentinel/analyze", json={
            "acquisition_before": {"id": "T1"},
            "detection_method": "temporal",
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("both acquisition_before and acquisition_after", response.get_json()["error"])
        self.assertIn("both image_before and image_after", response.get_json()["error"])

    def test_legacy_baseline_requires_explicit_selection_and_reports_model(self):
        forest_app.temporal_model = None
        stage1 = ConstantLogitModel(10.0)
        forest_app.stage1_model = stage1
        response = self.client.post("/api/sentinel/analyze", json={
            "image_before": make_rgb_data_url((20, 40, 60)),
            "image_after": make_rgb_data_url((40, 60, 80)),
            "detection_method": "baseline",
        })
        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertEqual(result["detection_method"], "baseline")
        self.assertEqual(result["model_used"], "stage1_post_classification")
        self.assertNotIn("temporal_unet", result["checkpoint_info"])
        self.assertEqual(stage1.seen_shape, (1, 3, 256, 256))

    def test_legacy_unknown_method_does_not_fall_back_to_another_model(self):
        response = self.client.post("/api/sentinel/analyze", json={
            "image_before": make_rgb_data_url((20, 40, 60)),
            "image_after": make_rgb_data_url((40, 60, 80)),
            "detection_method": "automatic",
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("detection_method", response.get_json()["error"])

    def test_masked_resize_excludes_invalid_source_values_from_valid_output(self):
        rgb = np.zeros((16, 16, 3), dtype=np.float32)
        rgb[:] = (0.2, 0.8, 0.3)
        valid = np.ones((16, 16), dtype=bool)
        rgb[:, 8:] = (1.0, 0.0, 0.0)
        valid[:, 8:] = False
        _, resized_valid, resized = forest_app._resize_rgb_for_temporal_model(rgb, valid, (8, 8))
        self.assertTrue(resized_valid.any())
        expected = np.tile(np.array([0.2, 0.8, 0.3]), (int(resized_valid.sum()), 1))
        np.testing.assert_allclose(resized[resized_valid], expected, atol=1 / 255)
        self.assertTrue(np.all(resized[~resized_valid] == 0.0))

    def test_invalid_pixels_are_rejected_before_model_inference(self):
        temporal = ConstantLogitModel(10.0)
        forest_app.temporal_model = temporal
        forest_app.stage1_model = ConstantLogitModel(10.0)
        tiff_bytes = make_analysis_tiff(all_invalid=True)
        payload = {
            "acquisition_before": {"id": "T1", "datetime": "2023-01-01T10:00:00Z"},
            "acquisition_after": {"id": "T2", "datetime": "2024-01-01T10:00:00Z"},
            "bbox": [-62.0, -3.0, -61.9, -2.9],
        }
        with patch.object(forest_app, "get_sentinel_token", return_value=("test-token", None, 200)), \
             patch.object(forest_app.requests, "post", return_value=FakeResponse(tiff_bytes)):
            response = self.client.post("/api/sentinel/analyze", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("no mutually clear", response.get_json()["error"].lower())
        self.assertIsNone(temporal.seen_shape)

    def test_misaligned_selected_scene_grids_are_rejected(self):
        forest_app.temporal_model = ConstantLogitModel(0.0)
        forest_app.stage1_model = ConstantLogitModel(0.0)
        responses = [
            FakeResponse(make_analysis_tiff()),
            FakeResponse(make_analysis_tiff(shift=0.001)),
        ]
        payload = {
            "acquisition_before": {"id": "T1", "datetime": "2023-01-01T10:00:00Z"},
            "acquisition_after": {"id": "T2", "datetime": "2024-01-01T10:00:00Z"},
            "bbox": [-62.0, -3.0, -61.9, -2.9],
        }
        with patch.object(forest_app, "get_sentinel_token", return_value=("test-token", None, 200)), \
             patch.object(forest_app.requests, "post", side_effect=responses):
            response = self.client.post("/api/sentinel/analyze", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("same spatial grid", response.get_json()["error"])

    def test_invalid_date_order_rejected_before_satellite_request(self):
        forest_app.temporal_model = ConstantLogitModel(0)
        forest_app.stage1_model = ConstantLogitModel(0)
        response = self.client.post("/api/sentinel/analyze", json={
            "acquisition_before": {"id": "T2", "datetime": "2024-01-01T00:00:00Z"},
            "acquisition_after": {"id": "T1", "datetime": "2023-01-01T00:00:00Z"},
            "bbox": [-62, -3, -61.9, -2.9],
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("earlier", response.get_json()["error"])

    def test_missing_temporal_checkpoint_returns_actionable_service_error(self):
        forest_app.temporal_model = None
        forest_app.stage1_model = ConstantLogitModel(0)
        response = self.client.post("/api/sentinel/analyze", json={
            "acquisition_before": {"id": "T1", "datetime": "2023-01-01T00:00:00Z"},
            "acquisition_after": {"id": "T2", "datetime": "2024-01-01T00:00:00Z"},
            "bbox": [-62, -3, -61.9, -2.9],
        })
        self.assertEqual(response.status_code, 503)
        result = response.get_json()
        self.assertIn("Temporal U-Net checkpoint", result["error"])
        self.assertEqual(result["detection_method"], "temporal")
        self.assertEqual(result["model_used"], "temporal_unet_with_stage1_loss_gate")
        self.assertEqual(result["model_status"], "not run")
        self.assertIsNone(result["prediction_output"])


if __name__ == "__main__":
    unittest.main()
