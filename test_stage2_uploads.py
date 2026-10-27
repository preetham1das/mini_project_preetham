import io
import unittest

from PIL import Image
from werkzeug.datastructures import FileStorage

from app import decode_stage2_upload_pair


def encoded_image(size, color):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


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


if __name__ == "__main__":
    unittest.main()
