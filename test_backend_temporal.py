import asyncio
import base64
import io
import unittest

import numpy as np
import torch
from PIL import Image
from starlette.datastructures import UploadFile

try:
    import app as live_app
except ModuleNotFoundError as exc:
    if exc.name != "rasterio":
        raise
    live_app = None

from backend import main
from backend.sentinel_service import (
    geodesic_area_hectares,
    parse_bbox,
    predict_directional_deforestation,
    rgba_mask_to_data_uri,
)


class ForestByBrightness(torch.nn.Module):
    def forward(self, image):
        return (image[:, :1] - 0.5) * 20


class ConstantTemporal(torch.nn.Module):
    def __init__(self, logit):
        super().__init__()
        self.logit = logit

    def forward(self, image):
        return image[:, :1] * 0 + self.logit


class ProbabilityTemporal(torch.nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.logit = float(np.log(probability / (1.0 - probability)))

    def forward(self, image):
        return image[:, :1] * 0 + self.logit


class CapturingTemporal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, image):
        self.input = image.detach().clone()
        return image[:, :1] * 0 + 10


def image_upload(color, filename):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
    buffer.seek(0)
    return UploadFile(filename=filename, file=buffer)


class TestDirectionalTemporalInference(unittest.TestCase):
    def test_preprocessing_produces_float32_unit_range_tensor(self):
        image = Image.new("RGB", (4, 4), (128, 64, 255))
        tensor = main.preprocess_image(image)

        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tuple(tensor.shape), (1, 3, 256, 256))
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 128 / 255, places=6)

    def test_empty_temporal_prediction_falls_back_to_directional_gate(self):
        image_t1 = torch.ones((1, 3, 8, 8), dtype=torch.float32)
        image_t2 = torch.zeros((1, 3, 8, 8), dtype=torch.float32)

        result = predict_directional_deforestation(
            ForestByBrightness(),
            ConstantTemporal(-10),
            image_t1,
            image_t2,
            torch.device("cpu"),
            threshold=0.5,
        )

        self.assertTrue(result["used_logical_fallback"])
        self.assertEqual(int(result["final_mask"].sum()), 64)
        self.assertLess(result["temporal_probability_max"], 0.2)
        self.assertEqual(result["temporal_threshold_used"], 0.5)

    def test_directional_gate_rejects_afforestation(self):
        image_t1 = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        image_t2 = torch.ones((1, 3, 8, 8), dtype=torch.float32)

        result = predict_directional_deforestation(
            ForestByBrightness(),
            ConstantTemporal(10),
            image_t1,
            image_t2,
            torch.device("cpu"),
            threshold=0.5,
        )

        self.assertFalse(result["used_logical_fallback"])
        self.assertEqual(int(result["final_mask"].sum()), 0)

    def test_low_probability_falls_back_without_changing_requested_threshold(self):
        image_t1 = torch.ones((1, 3, 8, 8), dtype=torch.float32)
        image_t2 = torch.zeros((1, 3, 8, 8), dtype=torch.float32)

        result = predict_directional_deforestation(
            ForestByBrightness(),
            ProbabilityTemporal(0.15),
            image_t1,
            image_t2,
            torch.device("cpu"),
            threshold=0.5,
        )

        self.assertEqual(result["temporal_threshold_used"], 0.5)
        self.assertTrue(result["used_logical_fallback"])
        self.assertEqual(int(result["final_mask"].sum()), 64)

    def test_temporal_input_is_the_exact_nine_channel_pair_and_difference(self):
        image_t1 = torch.tensor([[[[0.8]], [[0.25]], [[0.1]]]], dtype=torch.float32)
        image_t2 = torch.tensor([[[[0.2]], [[0.5]], [[0.8]]]], dtype=torch.float32)
        temporal_model = CapturingTemporal()

        result = predict_directional_deforestation(
            ForestByBrightness(),
            temporal_model,
            image_t1,
            image_t2,
            torch.device("cpu"),
            threshold=0.5,
        )

        self.assertEqual(tuple(temporal_model.input.shape), (1, 9, 1, 1))
        expected = torch.tensor(
            [[[[0.8]], [[0.25]], [[0.1]], [[0.2]], [[0.5]], [[0.8]],
              [[0.6]], [[0.25]], [[0.7]]]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(temporal_model.input, expected)
        self.assertEqual(int(result["final_mask"].sum()), 1)

    def test_forest_probability_equal_to_point_five_is_forest(self):
        image_t1 = torch.full((1, 3, 2, 2), 0.5, dtype=torch.float32)
        image_t2 = torch.zeros((1, 3, 2, 2), dtype=torch.float32)

        result = predict_directional_deforestation(
            ForestByBrightness(),
            ConstantTemporal(10),
            image_t1,
            image_t2,
            torch.device("cpu"),
            threshold=0.5,
        )

        self.assertEqual(int(result["final_mask"].sum()), 4)

    def test_bbox_geodesic_area_and_validation(self):
        bbox = parse_bbox("-62.25,-3.50,-62.15,-3.40")
        self.assertGreater(geodesic_area_hectares(bbox), 0)
        with self.assertRaisesRegex(ValueError, "bbox"):
            parse_bbox("-62.25,-3.50,-62.30,-3.40")

    def test_rgba_mask_keeps_background_transparent(self):
        encoded = rgba_mask_to_data_uri(np.array([[1, 0]], dtype=np.uint8))
        payload = encoded.split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(payload))) as mask_image:
            self.assertEqual(mask_image.mode, "RGBA")
            self.assertEqual(mask_image.getpixel((0, 0)), (255, 0, 0, 180))
            self.assertEqual(mask_image.getpixel((1, 0))[3], 0)

    def test_temporal_endpoint_returns_rgba_mask_and_metrics(self):
        original_forest_model = main.forest_model
        original_temporal_model = main.temporal_model
        main.forest_model = ForestByBrightness()
        main.temporal_model = ConstantTemporal(-10)
        try:
            response = asyncio.run(
                main.predict_temporal(
                    before_image=image_upload("white", "t1.png"),
                    after_image=image_upload("black", "t2.png"),
                    sample_path_a=None,
                    sample_path_b=None,
                    threshold=0.5,
                    bbox="-62.25,-3.50,-62.15,-3.40",
                )
            )
        finally:
            main.forest_model = original_forest_model
            main.temporal_model = original_temporal_model

        self.assertTrue(response["success"], response)
        self.assertEqual(response["changed_pixels"], response["total_pixels"])
        self.assertEqual(response["deforestation_percentage"], 100.0)
        self.assertAlmostEqual(
            response["area_loss_hectares"], response["total_area_ha"], places=4
        )
        self.assertTrue(response["deforestation_mask"].startswith("data:image/png;base64,"))
        self.assertIn("temporal_unet", response["benchmark_metadata"])

        encoded = response["deforestation_mask"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as mask_image:
            self.assertEqual(mask_image.mode, "RGBA")
            self.assertEqual(mask_image.getpixel((0, 0)), (255, 0, 0, 180))

    @unittest.skipIf(live_app is None, "Live Flask app requires the rasterio dependency.")
    def test_live_fetch_route_uses_directional_temporal_inference_and_geodesic_area(self):
        original_forest_model = live_app.stage1_model
        original_temporal_model = live_app.temporal_model
        live_app.stage1_model = ForestByBrightness()
        live_app.temporal_model = ConstantTemporal(-10)

        def image_data_uri(color):
            buffer = io.BytesIO()
            Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

        try:
            response = live_app.app.test_client().post(
                "/api/fetch-satellite-data",
                json={
                    "image_before": image_data_uri("white"),
                    "image_after": image_data_uri("black"),
                    "bbox": [-62.25, -3.50, -62.15, -3.40],
                    "threshold": 0.5,
                },
            )
        finally:
            live_app.stage1_model = original_forest_model
            live_app.temporal_model = original_temporal_model

        result = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(result["success"], result)
        self.assertEqual(result["changed_pixels"], result["total_pixels"])
        self.assertEqual(result["deforestation_percentage"], 100.0)
        self.assertGreater(result["area_loss_hectares"], 0)
        self.assertAlmostEqual(result["area_loss_hectares"], result["total_area_ha"], places=4)
        encoded = result["deforestation_mask"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as mask_image:
            self.assertEqual(mask_image.getpixel((0, 0)), (255, 0, 0, 180))


if __name__ == "__main__":
    unittest.main()
