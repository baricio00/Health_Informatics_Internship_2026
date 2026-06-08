import unittest

import torch

from lems_ct.src.metrics.utils import calculate_dice_split


class MetricsNonContiguousTests(unittest.TestCase):
    def test_calculate_dice_split_accepts_non_contiguous_one_hot(self):
        labels = torch.zeros((2, 4, 4, 4), dtype=torch.long)
        labels[0, 0:2, 0:2, 0:2] = 1
        labels[1, 2:4, 2:4, 2:4] = 1

        pred = torch.nn.functional.one_hot(labels, num_classes=2).movedim(-1, 1)
        target = torch.nn.functional.one_hot(labels, num_classes=2).movedim(-1, 1)

        self.assertFalse(pred.is_contiguous())

        dice, _, _ = calculate_dice_split(pred, target, 2, block_size=16)

        self.assertTrue(torch.allclose(dice, torch.ones_like(dice), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
