"""Contract tests for the server-side Sentinel-1 numerical data workflow."""
import io
import json
import tarfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

import app as forest_app


class FakeResponse:
    def __init__(self, status_code=200, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = content

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


def make_process_archive(width=32, height=32):
    transform = from_bounds(500000, 8000000, 500320, 8000320, width, height)
    profile = {"driver": "GTiff", "width": width, "height": height,
               "count": 2, "dtype": "float32", "crs": "EPSG:32722",
               "transform": transform, "nodata": -9999.0}
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(np.stack([np.full((height, width), .12, np.float32),
                                np.full((height, width), .04, np.float32)]))
            dst.set_band_description(1, "VV")
            dst.set_band_description(2, "VH")
        tif_bytes = mem.read()
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        for name, payload in (("default.tif", tif_bytes),
                              ("userdata.json", json.dumps({"orbits": [{"tiles": [{"date": "2024-05-01T10:00:00Z"}]}]}).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return archive_bytes.getvalue()


def make_s2_upload():
    transform = from_bounds(500000, 8000000, 500320, 8000320, 32, 32)
    profile = {"driver": "GTiff", "width": 32, "height": 32,
               "count": 4, "dtype": "float32", "crs": "EPSG:32722",
               "transform": transform}
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(np.ones((4, 32, 32), np.float32))
            for index, description in enumerate(("B2", "B3", "B4", "B8"), 1):
                dst.set_band_description(index, description)
        return mem.read()


def make_unverified_s1_upload():
    transform = from_bounds(500000, 8000000, 500320, 8000320, 32, 32)
    profile = {"driver": "GTiff", "width": 32, "height": 32,
               "count": 2, "dtype": "float32", "crs": "EPSG:32722",
               "transform": transform, "nodata": -9999.0}
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(np.ones((2, 32, 32), np.float32) * .1)
            dst.set_band_description(1, "VV")
            dst.set_band_description(2, "VH")
            dst.update_tags(sentinel1_calibration_status="UNVERIFIED_FOR_KAGGLE_V2")
        return mem.read()


class Sentinel1RetrievalTests(unittest.TestCase):
    def setUp(self):
        forest_app.app.config["TESTING"] = True
        self.client = forest_app.app.test_client()

    def test_retrieval_requests_dual_polarization_and_returns_aligned_two_band_tiff(self):
        archive = make_process_archive()
        features = [{"properties": {"s1:polarization": ["VV", "VH"]}}]

        def fake_post(url, **kwargs):
            if "/catalog/" in url:
                return FakeResponse(payload={"features": features})
            return FakeResponse(content=archive)

        with patch.object(forest_app, "get_sentinel_token", return_value=("test-token", None, 200)), \
             patch.object(forest_app.requests, "post", side_effect=fake_post) as post:
            response = self.client.post("/api/kaggle-amazon/sentinel1", data={
                "sentinel2": (io.BytesIO(make_s2_upload()), "s2.tif"),
                "start_date": "2024-05-01", "end_date": "2024-05-31",
            }, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200, response.json if response.is_json else response.status_code)
        self.assertEqual(post.call_count, 2)
        process_call = post.call_args_list[1]
        self.assertEqual(process_call.args[0], "https://sh.dataspace.copernicus.eu/process/v1")
        payload = process_call.kwargs["json"]
        self.assertEqual(payload["input"]["data"][0]["dataFilter"]["polarization"], "DV")
        self.assertEqual(payload["input"]["data"][0]["processing"]["backCoeff"], "GAMMA0_ELLIPSOID")
        self.assertEqual(payload["output"]["width"], 32)
        metadata = json.loads(response.headers["X-Sentinel1-Metadata"])
        self.assertEqual(metadata["bands"], ["VV", "VH"])
        self.assertFalse(metadata["calibration_compatible_with_kaggle_v2"])
        with MemoryFile(response.data) as mem, mem.open() as src:
            self.assertEqual(src.count, 2)
            self.assertEqual(src.descriptions, ("VV", "VH"))
            self.assertEqual(src.crs.to_epsg(), 32722)
            self.assertEqual((src.width, src.height), (32, 32))
            self.assertEqual(src.tags()["backscatter_coefficient"], "GAMMA0_ELLIPSOID")
            self.assertEqual(src.tags()["units"], "LINEAR_POWER (unitless)")
            self.assertEqual(src.tags()["sentinel1_calibration_status"], "UNVERIFIED_FOR_KAGGLE_V2")
            np.testing.assert_allclose(src.read(1), .12)
            np.testing.assert_allclose(src.read(2), .04)

    def test_empty_catalog_returns_actionable_no_data_error_without_process_call(self):
        with patch.object(forest_app, "get_sentinel_token", return_value=("test-token", None, 200)), \
             patch.object(forest_app.requests, "post", return_value=FakeResponse(payload={"features": []})) as post:
            response = self.client.post("/api/kaggle-amazon/sentinel1", data={
                "sentinel2": (io.BytesIO(make_s2_upload()), "s2.tif"),
                "start_date": "2024-05-01", "end_date": "2024-05-31",
            }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 422)
        self.assertIn("No Sentinel-1 GRD acquisitions", response.json["error"])
        self.assertEqual(post.call_count, 1)

    def test_invalid_range_is_rejected_before_auth_or_network(self):
        with patch.object(forest_app, "get_sentinel_token") as auth:
            response = self.client.post("/api/kaggle-amazon/sentinel1", data={
                "sentinel2": (io.BytesIO(make_s2_upload()), "s2.tif"),
                "start_date": "2024-06-01", "end_date": "2024-05-01",
            }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Start date", response.json["error"])
        auth.assert_not_called()

    def test_retrieved_linear_power_raster_is_blocked_from_uncalibrated_v2_inference(self):
        with patch.object(forest_app, "get_kaggle_amazon_model", return_value=object()):
            response = self.client.post("/api/predict/kaggle-amazon", data={
                "sentinel1": (io.BytesIO(make_unverified_s1_upload()), "s1.tif"),
                "sentinel2": (io.BytesIO(make_s2_upload()), "s2.tif"),
                "threshold": "0.5",
            }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("not calibrated to the V2 training data", response.json["error"])


if __name__ == "__main__":
    unittest.main()
