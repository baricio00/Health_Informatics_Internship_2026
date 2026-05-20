import unittest

from scripts import UNETR as unetr


class UNETREntrypointTests(unittest.TestCase):
    def test_applies_unetr_defaults_from_roi_size(self):
        model_cfg = {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 2,
            "init_filters": 16,
            "blocks_down": [1, 2, 2, 4],
            "blocks_up": [1, 1, 1],
        }

        result = unetr.with_unetr_defaults(model_cfg, [128, 128, 128])

        self.assertEqual(result["img_size"], [128, 128, 128])
        self.assertEqual(result["feature_size"], 16)
        self.assertEqual(result["hidden_size"], 768)
        self.assertEqual(result["mlp_dim"], 3072)
        self.assertEqual(result["num_heads"], 12)
        self.assertEqual(result["proj_type"], "perceptron")
        self.assertEqual(result["norm_name"], "instance")
        self.assertEqual(result["res_block"], True)
        self.assertEqual(result["dropout_rate"], 0.0)
        self.assertEqual(result["in_channels"], 1)
        self.assertEqual(result["out_channels"], 2)

    def test_filters_segresnet_only_keys_from_unetr_kwargs(self):
        model_cfg = unetr.with_unetr_defaults(
            {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 2,
                "init_filters": 16,
                "blocks_down": [1, 2, 2, 4],
                "blocks_up": [1, 1, 1],
                "feature_size": 32,
                "dropout_rate": 0.1,
            },
            [96, 96, 96],
        )

        result = unetr.unetr_kwargs_for_signature(
            model_cfg,
            {
                "in_channels",
                "out_channels",
                "img_size",
                "feature_size",
                "hidden_size",
                "dropout_rate",
                "spatial_dims",
            },
        )

        self.assertEqual(
            result,
            {
                "in_channels": 1,
                "out_channels": 2,
                "img_size": (96, 96, 96),
                "feature_size": 32,
                "hidden_size": 768,
                "dropout_rate": 0.1,
                "spatial_dims": 3,
            },
        )
        self.assertNotIn("init_filters", result)
        self.assertNotIn("blocks_down", result)
        self.assertNotIn("blocks_up", result)

    def test_arg_parser_defaults_to_fold_zero(self):
        parser = unetr.build_arg_parser()

        args = parser.parse_args(
            [
                "--input_data",
                "azureml:LEMS-CT-NIfTI:1",
                "--output_model",
                "outputs",
            ]
        )

        self.assertEqual(args.fold, 0)


if __name__ == "__main__":
    unittest.main()
