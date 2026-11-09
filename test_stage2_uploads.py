import io
import unittest
from unittest.mock import patch

from PIL import Image
from werkzeug.datastructures import FileStorage
import torch

import app
from app import decode_stage2_upload_pair


def encoded_image(size, color):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class FirstChannelForestModel(torch.nn.Module):
    def forward(self, image):
        return image[:, :1] * 2 - 1


class TestStage2UploadPair(unittest.TestCase):
    def test_accepts_matching_dimensions(self):
        image_t1, image_t2 = decode_stage2_upload_pair(
            FileStorage(stream=encoded_image((32, 24), "green")),
            FileStorage(stream=encoded_image((32, 24), "brown")),
        )

        self.assertEqual(image_t1.size, (32, 24))
        self.assertEqual(image_t2.size, (32, 24))
        self.assertEqual(image_t1.mode, "RGB")
        self.assertEqual(image_t2.mode, "RGB")

    def test_rejects_different_dimensions(self):
        with self.assertRaisesRegex(ValueError, "identical dimensions"):
            decode_stage2_upload_pair(
                FileStorage(stream=encoded_image((32, 24), "green")),
                FileStorage(stream=encoded_image((24, 32), "brown")),
            )

    def test_rejects_invalid_image(self):
        with self.assertRaisesRegex(ValueError, "Could not read"):
            decode_stage2_upload_pair(
                FileStorage(stream=io.BytesIO(b"not an image")),
                FileStorage(stream=encoded_image((32, 24), "brown")),
            )

    def test_stage2_endpoint_applies_forest_loss_rule(self):
        with patch.object(app, "stage1_model", FirstChannelForestModel()):
            response = app.app.test_client().post(
                "/api/predict/stage2",
                data={
                    "file_t1": (encoded_image((32, 24), "white"), "t1.png"),
                    "file_t2": (encoded_image((32, 24), "black"), "t2.png"),
                    "threshold": "0.5",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["deforestation_pixels"], 256 * 256)
        self.assertEqual(result["deforestation_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
