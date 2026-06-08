import unittest

import torch

from scripts import lcc_postprocessing as lcc


class LCCPostprocessingTests(unittest.TestCase):
    def test_keep_largest_cc_after_argmax_uses_cpu_scipy_and_preserves_device(self):
        mask = torch.zeros((1, 5, 5, 5), dtype=torch.long)
        mask[0, 0:3, 0:3, 0:3] = 1
        mask[0, 4, 4, 4] = 1

        cleaned = lcc.keep_largest_cc_after_argmax(mask)

        self.assertEqual(cleaned.device, mask.device)
        self.assertEqual(cleaned.dtype, torch.uint8)
        self.assertEqual(int(cleaned.sum().item()), 27)
        self.assertEqual(int(cleaned[0, 4, 4, 4].item()), 0)

    def test_lcc_one_hot_after_argmax_returns_one_hot_cleaned_mask(self):
        logits = torch.zeros((1, 2, 5, 5, 5), dtype=torch.float32)
        logits[:, 0] = 1.0
        logits[0, 1, 0:3, 0:3, 0:3] = 3.0
        logits[0, 1, 4, 4, 4] = 3.0

        cleaned = lcc.lcc_one_hot_after_argmax(logits, out_channels=2)

        self.assertEqual(cleaned.shape, logits.shape)
        self.assertEqual(cleaned.dtype, torch.uint8)
        self.assertEqual(int(cleaned[0, 1].sum().item()), 27)
        self.assertEqual(int(cleaned[0, 1, 4, 4, 4].item()), 0)
        self.assertTrue(torch.all(cleaned.sum(dim=1) == 1))


if __name__ == "__main__":
    unittest.main()
