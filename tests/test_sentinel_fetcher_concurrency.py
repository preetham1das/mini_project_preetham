import os
import unittest
from datetime import datetime, timezone
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend import sentinel_fetcher


class FakeItem:
    def __init__(self):
        self.properties = {
            "eo:cloud_cover": 5,
            "datetime": "2024-06-15T12:00:00Z",
        }
        self.datetime = datetime(2024, 6, 15, 12, tzinfo=timezone.utc)
        self.assets = {
            key: SimpleNamespace(href=key)
            for key in sentinel_fetcher.ASSET_KEYS
        }


class FakeSearch:
    def __init__(self, item):
        self.item = item

    def items(self):
        return [self.item]


class FakeCatalog:
    def __init__(self, item):
        self.item = item

    def search(self, **kwargs):
        return FakeSearch(self.item)


class SentinelFetcherConcurrencyTests(unittest.TestCase):
    def test_reads_five_bands_concurrently_and_sets_gdal_timeouts(self):
        item = FakeItem()
        barrier = Barrier(5)

        def read_band(_item, asset_key, _bbox, size, _interpolation):
            barrier.wait(timeout=3)
            value = 4 if asset_key == "SCL" else 10_000
            return np.full(size, value, dtype=np.uint16)

        with (
            patch.object(
                sentinel_fetcher.Client,
                "open",
                return_value=FakeCatalog(item),
            ),
            patch.object(sentinel_fetcher.planetary_computer, "sign_inplace"),
            patch.object(
                sentinel_fetcher,
                "_read_resized_band",
                side_effect=read_band,
            ),
        ):
            result = sentinel_fetcher.fetch_s2_patch(
                [-62.0, -3.0, -61.9, -2.9],
                "2024-06-15",
                size=(2, 2),
            )

        self.assertEqual(os.environ["GDAL_HTTP_TIMEOUT"], "15")
        self.assertEqual(os.environ["GDAL_HTTP_MAX_RETRY"], "2")
        self.assertEqual(os.environ["GDAL_DISABLE_READDIR_ON_OPEN"], "EMPTY_DIR")
        self.assertTrue(np.all(result["valid_mask"] == 1))
        self.assertTrue(np.all(result["red"] == 1.0))
        self.assertEqual(result["scene_date"], "2024-06-15")


if __name__ == "__main__":
    unittest.main()
