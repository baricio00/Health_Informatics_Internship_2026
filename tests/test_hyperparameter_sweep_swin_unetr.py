import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from scripts import hyperparameter_sweep_single_fold as sweep


class HyperparameterSweepSwinUNETRTests(unittest.TestCase):
    def base_args(self, sweep_dir):
        return SimpleNamespace(
            lrs=["3e-4"],
            weight_decays=["1e-5"],
            roi_sizes=["96"],
            losses=["dice_focal"],
            target_spacings=["1.0"],
            max_iterations=6000,
            fold=0,
            large_roi_threshold=96,
            disable_large_roi_memory_overrides=False,
            large_roi_train_batch_size=1,
            large_roi_accumulation_steps=4,
            large_roi_sw_batch_size=1,
            large_roi_disable_ema=False,
            disable_hausdorff_memory_overrides=False,
            hausdorff_train_batch_size=1,
            hausdorff_accumulation_steps=4,
            hausdorff_downsample=1,
            hausdorff_validation_loss="dice",
            hausdorff_sw_batch_size=1,
            hausdorff_disable_ema=True,
            extra_overrides=[
                "--pretrained_swin_encoder",
                "weights/model_swinvit.pt",
            ],
            checkpoint=None,
            python_executable="python",
            train_script="scripts/swin_UNETR.py",
            input_data="/tmp/input",
            output_model=str(sweep_dir),
            split_csv="data/cv_splits.csv",
            config="config/train_config.yaml",
            device="cuda",
            stop_on_trial_error=False,
            limit=None,
        )

    def test_azure_command_uses_configured_swin_train_script(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self.base_args(Path(temp_dir))
            jobs = sweep.build_jobs(args, Path(temp_dir))

        self.assertEqual(len(jobs), 1)
        self.assertIn("python scripts/swin_UNETR.py", jobs[0]["azure_command"])
        self.assertNotIn("python scripts/train_job_only_dice.py", jobs[0]["azure_command"])
        self.assertIn(
            "--pretrained_swin_encoder weights/model_swinvit.pt",
            jobs[0]["azure_command"],
        )

    def test_azure_and_local_commands_include_checkpoint_when_finetuning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self.base_args(Path(temp_dir))
            args.checkpoint = (
                "azureml://example/checkpoints/fold_0/best_metric_model.pth"
            )
            jobs = sweep.build_jobs(args, Path(temp_dir))

        self.assertIn(
            "--checkpoint ${{inputs.checkpoint_file}}",
            jobs[0]["azure_command"],
        )
        self.assertIn("--checkpoint", jobs[0]["command"])
        self.assertIn(
            "azureml://example/checkpoints/fold_0/best_metric_model.pth",
            jobs[0]["command"],
        )
        self.assertEqual(
            jobs[0]["checkpoint"],
            "azureml://example/checkpoints/fold_0/best_metric_model.pth",
        )

    def test_azure_backend_does_not_require_local_input_data(self):
        argv = [
            "hyperparameter_sweep_single_fold.py",
            "--backend",
            "azure",
            "--fold",
            "0",
            "--train_script",
            "scripts/swin_UNETR.py",
            "--pretrained_swin_encoder",
            "weights/model_swinvit.pt",
        ]

        with patch("sys.argv", argv):
            args = sweep.parse_args()

        self.assertEqual(args.input_data, sweep.DEFAULT_AZURE_INPUT_DATA)
        self.assertEqual(args.extra_overrides, ["--pretrained_swin_encoder", "weights/model_swinvit.pt"])


if __name__ == "__main__":
    unittest.main()
