import unittest

from scripts import submit_swin_unetr_finetune_queue as queue


class SwinUNETRFinetuneQueueSubmitterTests(unittest.TestCase):
    def test_defaults_target_swin_finetune_job(self):
        args = queue.parse_args(
            [
                "--checkpoint_path",
                "azureml://example/fold_{fold}/best_metric_model.pth",
            ]
        )

        self.assertEqual(args.job_file, "jobs/swin_unetr_finetune_job.yml")
        self.assertEqual(args.compute, "azureml:clusterprdwe-g-t2-vzhst6")

    def test_build_job_create_command_sets_fold_and_checkpoint(self):
        command = queue.build_job_create_command(
            job_file="jobs/swin_unetr_finetune_job.yml",
            fold=1,
            checkpoint_path="azureml://example/fold_1/best_metric_model.pth",
            az_executable="az",
        )

        self.assertIn("inputs.fold=1", command)
        self.assertIn(
            "inputs.checkpoint_file.path=azureml://example/fold_1/best_metric_model.pth",
            command,
        )

    def test_formats_checkpoint_template_with_fold(self):
        result = queue.checkpoint_for_fold(
            "azureml://example/fold_{fold}/best_metric_model.pth",
            4,
        )

        self.assertEqual(result, "azureml://example/fold_4/best_metric_model.pth")

    def test_rejects_unresolved_checkpoint_placeholders(self):
        with self.assertRaises(ValueError) as context:
            queue.checkpoint_for_fold(
                "azureml://datastores/workspaceblobstore/paths/azureml/"
                "<SOURCE_JOB_NAME>/output_model/fold_{fold}/best_metric_model.pth",
                0,
            )

        self.assertIn("<SOURCE_JOB_NAME>", str(context.exception))


if __name__ == "__main__":
    unittest.main()
