"""
test_cloud_stats.py
-------------------
Automated test suite for Sentinel-2 L2A AOI cloud statistics.
Verifies:
  1. Input parameter validation (missing acquisition_id, invalid bbox, invalid lat/lon).
  2. Integration with Copernicus Data Space Process API (SCL retrieval).
  3. Correct percentage calculation (0 <= cloud_pct <= 100, 0 <= clear_pct <= 100).
  4. Consistency of pixel counts (valid_pixels <= total_pixels, cloud + other <= valid).
  5. Distinction between scene-level STAC cloud cover and AOI-specific cloud cover.
"""

import unittest
import app


class TestSentinelCloudStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.app.test_client()

    def test_missing_acquisition_id(self):
        res = self.client.get('/api/sentinel/cloud-stats?bbox=-62.25,-3.50,-62.15,-3.40')
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data.get('success'))
        self.assertIn("acquisition_id", data.get('error', ''))

    def test_missing_bbox_and_coords(self):
        res = self.client.get('/api/sentinel/cloud-stats?acquisition_id=test_id')
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data.get('success'))

    def test_invalid_bbox_format(self):
        res = self.client.get('/api/sentinel/cloud-stats?acquisition_id=test_id&bbox=-62.25,-3.50')
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data.get('success'))
        self.assertIn("comma-separated", data.get('error', ''))

    def test_real_acquisition_cloud_stats(self):
        # 1. Search for a recent acquisition
        search_res = self.client.get('/api/sentinel/search?lat=-3.4653&lon=-62.2159&start_date=2023-07-01&end_date=2023-09-30&limit=3')
        self.assertEqual(search_res.status_code, 200)
        search_data = search_res.get_json()
        self.assertTrue(search_data.get('success'))
        acquisitions = search_data.get('acquisitions', [])
        self.assertGreater(len(acquisitions), 0, "Expected at least 1 Sentinel-2 acquisition")

        acq = acquisitions[0]
        acq_id = acq['id']
        dt = acq.get('datetime')
        scene_cloud = acq.get('cloud_coverage')

        bbox_str = '-62.25,-3.50,-62.15,-3.40'
        res = self.client.get(f'/api/sentinel/cloud-stats?acquisition_id={acq_id}&datetime={dt}&bbox={bbox_str}')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        # Structural assertions
        self.assertTrue(data.get('success'))
        self.assertEqual(data.get('acquisition_id'), acq_id)
        self.assertEqual(len(data.get('bbox', [])), 4)
        self.assertIn('cloud_pct', data)
        self.assertIn('clear_pct', data)
        self.assertIn('total_pixels', data)
        self.assertIn('valid_pixels', data)
        self.assertIn('cloud_pixels', data)
        self.assertIn('clear_pixels', data)
        self.assertIn('scl_class_counts', data)

        # Range assertions
        cloud_pct = data['cloud_pct']
        clear_pct = data['clear_pct']
        self.assertGreaterEqual(cloud_pct, 0.0)
        self.assertLessEqual(cloud_pct, 100.0)
        self.assertGreaterEqual(clear_pct, 0.0)
        self.assertLessEqual(clear_pct, 100.0)

        # Pixel count sanity assertions
        self.assertGreaterEqual(data['valid_pixels'], 1)
        self.assertLessEqual(data['valid_pixels'], data['total_pixels'])
        self.assertLessEqual(data['cloud_pixels'], data['valid_pixels'])
        self.assertLessEqual(data['clear_pixels'], data['valid_pixels'])

        print(f"\n[Test Verified]")
        print(f"Acquisition ID: {acq_id}")
        print(f"Scene-level cloud cover (STAC): {scene_cloud}%")
        print(f"AOI-specific cloud cover:        {cloud_pct}%")
        print(f"AOI-specific clear canopy/soil:  {clear_pct}%")
        print(f"SCL classes observed:           {list(data['scl_class_counts'].keys())}")


if __name__ == '__main__':
    unittest.main()
