import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOB_FILE = PROJECT_ROOT / "jobs" / "unetr_finetune_job.yml"


class UNETRFinetuneJobConfigTests(unittest.TestCase):
    def test_unetr_finetune_job_loads_checkpoint_without_resuming(self):
        job_text = JOB_FILE.read_text()

        self.assertIn("python scripts/UNETR.py", job_text)
        self.assertIn("--checkpoint ${{inputs.checkpoint_file}}", job_text)
        self.assertNotIn("--resume", job_text)

    def test_unetr_finetune_checkpoint_path_is_not_placeholder(self):
        job_text = JOB_FILE.read_text()

        self.assertNotIn("path/to", job_text)
        self.assertNotIn("<SOURCE_JOB_NAME>", job_text)

    def test_unetr_finetune_downloads_checkpoint_file(self):
        job_text = JOB_FILE.read_text()

        self.assertIn("checkpoint_file:", job_text)
        self.assertIn("mode: download", job_text)

    def test_unetr_finetune_job_uses_gpu_cuda(self):
        job_text = JOB_FILE.read_text()

        self.assertIn("compute: azureml:clusterprdwe-vzhst6", job_text)
        self.assertIn("--device cuda", job_text)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True", job_text)


if __name__ == "__main__":
    unittest.main()
