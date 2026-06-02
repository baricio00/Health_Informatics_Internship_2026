import unittest

from scripts import submit_swin_unetr_finetune_pipeline as pipeline


class SwinUNETRFinetunePipelineSubmitterTests(unittest.TestCase):
    def test_defaults_to_all_five_folds(self):
        args = pipeline.parse_args(
            [
                "--checkpoint_path",
                "azureml://example/output_model/fold_{fold}/best_metric_model.pth",
                "--dry_run",
            ]
        )

        self.assertEqual(args.folds, [0, 1, 2, 3, 4])
        self.assertEqual(args.azure_compute, "azureml:clusterprdwe-g-t2-vzhst6")

    def test_builds_one_child_spec_per_fold_from_checkpoint_template(self):
        args = pipeline.parse_args(
            [
                "--checkpoint_path",
                "azureml://example/output_model/fold_{fold}/best_metric_model.pth",
                "--dry_run",
            ]
        )

        specs = pipeline.build_fold_specs(args)

        self.assertEqual([spec.fold for spec in specs], [0, 1, 2, 3, 4])
        self.assertEqual(
            specs[3].checkpoint_path,
            "azureml://example/output_model/fold_3/best_metric_model.pth",
        )
        self.assertIn("--fold 3", specs[3].command)
        self.assertIn("--checkpoint ${{inputs.checkpoint_file}}", specs[3].command)

    def test_builds_checkpoint_paths_from_per_fold_source_jobs(self):
        args = pipeline.parse_args(
            [
                "--checkpoint_job_names",
                "job0",
                "job1",
                "job2",
                "job3",
                "job4",
                "--dry_run",
            ]
        )

        specs = pipeline.build_fold_specs(args)

        self.assertEqual(
            specs[4].checkpoint_path,
            "azureml://datastores/workspaceblobstore/paths/azureml/"
            "job4/output_model/fold_4/best_metric_model.pth",
        )

    def test_rejects_mismatched_checkpoint_job_name_count(self):
        args = pipeline.parse_args(
            [
                "--checkpoint_job_names",
                "job0",
                "job1",
                "--folds",
                "0",
                "1",
                "2",
                "--dry_run",
            ]
        )

        with self.assertRaises(ValueError) as context:
            pipeline.build_fold_specs(args)

        self.assertIn("one checkpoint job name per fold", str(context.exception))


if __name__ == "__main__":
    unittest.main()
