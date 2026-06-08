import unittest
import tempfile
from pathlib import Path

from scripts import submit_segresnet_cv_pipeline as pipeline


class SegResNetCVPipelineSubmitterTests(unittest.TestCase):
    def test_defaults_to_all_five_folds_and_qc_split(self):
        args = pipeline.parse_args(["--dry_run"])

        self.assertEqual(args.folds, [0, 1, 2, 3, 4])
        self.assertEqual(args.split_csv, "data/cv_splits_qc.csv")
        self.assertEqual(args.azure_compute, "azureml:clusterprdwe-g-t2-vzhst6")

    def test_training_template_uses_gpu_compute(self):
        job_text = Path("jobs/train_job.yml").read_text()

        self.assertIn("compute: azureml:clusterprdwe-g-t2-vzhst6", job_text)

    def test_builds_training_spec_per_fold(self):
        args = pipeline.parse_args(["--dry_run"])

        specs = pipeline.build_training_specs(args)

        self.assertEqual([spec.fold for spec in specs], [0, 1, 2, 3, 4])
        self.assertIn("--fold 3", specs[3].command)
        self.assertIn("--split_csv data/cv_splits_qc.csv", specs[3].command)
        self.assertIn("python scripts/train_job.py", specs[3].command)

    def test_inference_command_uses_fold_outputs_and_qc_split(self):
        args = pipeline.parse_args(["--dry_run"])

        command = pipeline.inference_command(args)

        self.assertIn("--split_csv data/cv_splits_qc.csv", command)
        self.assertIn("--w0 ${{inputs.w0}}", command)
        self.assertIn("--w4 ${{inputs.w4}}", command)

    def test_prepares_minimal_code_bundle_without_heavy_local_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = pipeline.parse_args(["--dry_run", "--output_dir", tmp])

            code_dir = pipeline.prepare_code_bundle(args)

            self.assertEqual(code_dir, Path(tmp) / "azure_code")
            self.assertTrue((code_dir / "scripts" / "train_job.py").exists())
            self.assertTrue((code_dir / "jobs" / "inference_job.py").exists())
            self.assertTrue((code_dir / "lems_ct" / "src" / "utils" / "data.py").exists())
            self.assertTrue((code_dir / "config" / "train_config.yaml").exists())
            self.assertTrue((code_dir / "data" / "cv_splits_qc.csv").exists())
            self.assertFalse((code_dir / "notebooks").exists())
            self.assertFalse((code_dir / "weights").exists())
            self.assertFalse((code_dir / "outputs").exists())
            self.assertFalse((code_dir / "mlruns").exists())


if __name__ == "__main__":
    unittest.main()
