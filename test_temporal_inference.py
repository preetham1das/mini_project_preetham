import unittest

import torch

from src.temporal_inference import (
    calculate_deforestation_mask,
    predict_temporal_forest_masks,
)


class FirstChannelForestModel(torch.nn.Module):
    def forward(self, image):
        return image[:, :1] * 2 - 1


class TestTemporalInference(unittest.TestCase):
    def test_deforestation_requires_forest_at_t1_and_nonforest_at_t2(self):
        image_t1 = torch.tensor(
            [[[[1, 1, 0, 0]], [[1, 1, 0, 0]], [[1, 1, 0, 0]]]],
            dtype=torch.float32,
        )
        image_t2 = torch.tensor(
            [[[[0, 1, 0, 1]], [[0, 1, 0, 1]], [[0, 1, 0, 1]]]],
            dtype=torch.float32,
        )

        mask_t1, mask_t2, deforestation = predict_temporal_forest_masks(
            FirstChannelForestModel(),
            image_t1,
            image_t2,
            torch.device("cpu"),
            threshold=0.5,
        )

        self.assertEqual(mask_t1.tolist(), [[1, 1, 0, 0]])
        self.assertEqual(mask_t2.tolist(), [[0, 1, 0, 1]])
        self.assertEqual(deforestation.tolist(), [[1, 0, 0, 0]])

    def test_rejects_misaligned_forest_masks(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            calculate_deforestation_mask([[1, 0]], [[0]])


if __name__ == "__main__":
    unittest.main()
