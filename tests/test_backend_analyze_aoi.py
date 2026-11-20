import base64
import io
import unittest
from threading import Barrier
from unittest.mock import patch

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image

import backend.main as backend_main


class DirectionalLogitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, images):
        logit = 10.0 if self.calls == 0 else -10.0
        self.calls += 1
        return torch.full(
            (images.shape[0], 1, images.shape[2], images.shape[3]),
            logit,
            device=images.device,
        )


def _make_patch(scene_date, red, nir, forest_region=False):
    shape = (backend_main.IMAGE_SIZE, backend_main.IMAGE_SIZE)
    red_band = np.full(shape, red, dtype=np.float32)
    nir_band = np.full(shape, nir, dtype=np.float32)
    if forest_region:
        red_band[80:90, 80:90] = 0.1
        nir_band[80:90, 80:90] = 0.8
    return {
        "red": red_band,
        "green": np.full(shape, 0.1, dtype=np.float32),
        "blue": np.full(shape, 0.1, dtype=np.float32),
        "nir": nir_band,
        "valid_mask": np.ones(shape, dtype=np.uint8),
        "scene_date": scene_date,
    }


class AnalyzeAOIEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(backend_main.app)

    def test_analyze_aoi_applies_directional_gate_and_returns_rgba_png(self):
        t1 = _make_patch("2020-01-01", 0.1, 0.1, forest_region=True)
        t2 = _make_patch("2024-01-01", 0.5, 0.6)
        bbox = [-62.0, -3.0, -61.9, -2.9]
        fetch_barrier = Barrier(2)

        def fetch_patch(request_bbox, target_date, **kwargs):
            fetch_barrier.wait(timeout=3)
            return t1 if target_date == "2020-01-01" else t2

        with (
            patch.object(backend_main, "forest_model", DirectionalLogitModel()),
            patch.object(
                backend_main,
                "fetch_s2_patch",
                side_effect=fetch_patch,
            ) as fetch,
        ):
            response = self.client.post(
                "/api/analyze-aoi",
                json={
                    "bbox": bbox,
                    "t1_date": "2020-01-01",
                    "t2_date": "2024-01-01",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["deforested_pixels"], 100)
        self.assertEqual(result["t1_scene_date"], "2020-01-01")
        self.assertEqual(result["t2_scene_date"], "2024-01-01")
        self.assertEqual(result["bbox"], bbox)
        self.assertEqual(fetch.call_count, 2)
        expected_area = backend_main.geodesic_area_hectares(tuple(bbox)) * 100 / (
            backend_main.IMAGE_SIZE**2
        )
        self.assertAlmostEqual(result["area_hectares"], expected_area)
        self.assertEqual(fetch.call_args_list[0].args[:2], (bbox, "2020-01-01"))
        self.assertEqual(fetch.call_args_list[1].args[:2], (bbox, "2024-01-01"))

        image_bytes = base64.b64decode(result["overlay_png"].split(",", 1)[1])
        with Image.open(io.BytesIO(image_bytes)) as overlay:
            rgba = np.asarray(overlay.convert("RGBA"))
        self.assertEqual(tuple(rgba[84, 84]), (239, 68, 68, 195))
        self.assertEqual(tuple(rgba[0, 0]), (0, 0, 0, 0))

    def test_analyze_aoi_removes_components_smaller_than_twenty_pixels(self):
        t1 = _make_patch("2020-01-01", 0.1, 0.1, forest_region=True)
        t1["nir"][80:90, 80:90] = 0.1
        t1["nir"][80:84, 80:84] = 0.8
        t2 = _make_patch("2024-01-01", 0.5, 0.6)

        def fetch_patch(request_bbox, target_date, **kwargs):
            return t1 if target_date == "2020-01-01" else t2

        with (
            patch.object(backend_main, "forest_model", DirectionalLogitModel()),
            patch.object(
                backend_main,
                "fetch_s2_patch",
                side_effect=fetch_patch,
            ),
        ):
            response = self.client.post(
                "/api/analyze-aoi",
                json={
                    "bbox": [-62.0, -3.0, -61.9, -2.9],
                    "t1_date": "2020-01-01",
                    "t2_date": "2024-01-01",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["deforested_pixels"], 0)

    def test_analyze_aoi_rejects_non_increasing_bbox(self):
        response = self.client.post(
            "/api/analyze-aoi",
            json={
                "bbox": [-62.0, -3.0, -62.0, -2.9],
                "t1_date": "2020-01-01",
                "t2_date": "2024-01-01",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_analyze_aoi_rejects_bbox_larger_than_point_fifteen_degrees(self):
        with patch.object(backend_main, "fetch_s2_patch") as fetch:
            response = self.client.post(
                "/api/analyze-aoi",
                json={
                    "bbox": [-62.0, -3.0, -61.0, -2.0],
                    "t1_date": "2020-01-01",
                    "t2_date": "2024-01-01",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("0.15 degrees", response.json()["detail"])
        fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
