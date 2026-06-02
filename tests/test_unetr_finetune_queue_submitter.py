import unittest

from scripts import submit_unetr_finetune_queue as queue


class UNETRFinetuneQueueSubmitterTests(unittest.TestCase):
    def test_build_job_create_command_sets_fold_and_checkpoint(self):
        command = queue.build_job_create_command(
            job_file="jobs/unetr_finetune_job.yml",
            fold=2,
            checkpoint_path="azureml://example/fold_2/best_metric_model.pth",
            az_executable="az",
        )

        self.assertEqual(command[:5], ["az", "ml", "job", "create", "--file"])
        self.assertIn("inputs.fold=2", command)
        self.assertIn(
            "inputs.checkpoint_file.path=azureml://example/fold_2/best_metric_model.pth",
            command,
        )

    def test_formats_checkpoint_template_with_fold(self):
        result = queue.checkpoint_for_fold(
            "azureml://example/fold_{fold}/best_metric_model.pth",
            3,
        )

        self.assertEqual(result, "azureml://example/fold_3/best_metric_model.pth")

    def test_builds_checkpoint_path_from_source_job_name(self):
        result = queue.checkpoint_path_from_job_name("cyan_planet_yhcvz8pgc1", 0)

        self.assertEqual(
            result,
            "azureml://datastores/workspaceblobstore/paths/azureml/"
            "cyan_planet_yhcvz8pgc1/output_model/fold_0/best_metric_model.pth",
        )

    def test_parse_args_accepts_source_job_name_instead_of_checkpoint_path(self):
        args = queue.parse_args(["--checkpoint_job_name", "cyan_planet_yhcvz8pgc1"])

        self.assertIsNone(args.checkpoint_path)
        self.assertEqual(args.checkpoint_job_name, "cyan_planet_yhcvz8pgc1")

    def test_rejects_unresolved_checkpoint_placeholders(self):
        with self.assertRaises(ValueError) as context:
            queue.checkpoint_for_fold(
                "azureml://datastores/workspaceblobstore/paths/azureml/"
                "<SOURCE_JOB_NAME>/output_model/fold_{fold}/best_metric_model.pth",
                0,
            )

        self.assertIn("<SOURCE_JOB_NAME>", str(context.exception))

    def test_filters_active_jobs_on_compute(self):
        jobs = [
            {"name": "a", "status": "Running", "compute": "azureml:gpu"},
            {"name": "b", "status": "Completed", "compute": "azureml:gpu"},
            {"name": "c", "status": "Queued", "compute": "gpu"},
            {"name": "d", "status": "Running", "compute": "azureml:cpu"},
        ]

        active = queue.active_jobs_on_compute(jobs, "azureml:gpu")

        self.assertEqual([job["name"] for job in active], ["a", "c"])


if __name__ == "__main__":
    unittest.main()
