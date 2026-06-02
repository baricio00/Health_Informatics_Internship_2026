import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOB_FILE = PROJECT_ROOT / "jobs" / "swin_unetr_train_job.yml"


class SwinUNETRJobConfigTests(unittest.TestCase):
    def test_swin_job_requires_gpu_cuda(self):
        job_text = JOB_FILE.read_text()

        self.assertIn("compute: azureml:clusterprdwe-g-t2-vzhst6", job_text)
        self.assertIn("python scripts/swin_UNETR.py", job_text)
        self.assertIn("--device cuda", job_text)
        self.assertIn("--pretrained_swin_encoder weights/model_swinvit.pt", job_text)

    def test_swin_job_uses_memory_safe_batching(self):
        job_text = JOB_FILE.read_text()

        self.assertIn("training.train_batch_size=1", job_text)
        self.assertIn("training.accumulation_steps=4", job_text)
        self.assertIn("inference.sw_batch_size=1", job_text)


if __name__ == "__main__":
    unittest.main()
