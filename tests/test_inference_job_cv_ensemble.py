import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


def stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


dependency_stubs = {
    "torch": stub_module(
        "torch",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        device=lambda value: value,
        inference_mode=lambda: mock.MagicMock(),
        amp=types.SimpleNamespace(autocast=lambda *args, **kwargs: mock.MagicMock()),
        softmax=lambda value, dim: value,
    ),
    "monai.data": stub_module(
        "monai.data",
        DataLoader=object,
        Dataset=object,
        MetaTensor=object,
        decollate_batch=lambda value: value,
    ),
    "monai.inferers": stub_module(
        "monai.inferers", sliding_window_inference=lambda *args, **kwargs: None
    ),
    "monai.networks.nets": stub_module("monai.networks.nets", SegResNet=object),
    "monai.transforms": stub_module(
        "monai.transforms",
        AsDiscrete=object,
        AsDiscreted=object,
        Compose=object,
        CropForegroundd=object,
        EnsureChannelFirstd=object,
        Invertd=object,
        LoadImaged=object,
        Orientationd=object,
        SaveImaged=object,
        ScaleIntensityRanged=object,
        Spacingd=object,
    ),
    "omegaconf": stub_module(
        "omegaconf",
        OmegaConf=types.SimpleNamespace(load=lambda *args, **kwargs: None),
    ),
}


with mock.patch.dict(sys.modules, dependency_stubs):
    from scripts import inference_job


class InferenceJobCVEnsembleTests(unittest.TestCase):
    def test_collect_checkpoint_paths_accepts_one_root_per_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = []
            for fold in range(5):
                root = Path(tmp) / f"fold_output_{fold}"
                checkpoint = root / f"fold_{fold}" / "best_metric_model.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"checkpoint")
                roots.append(str(root))

            paths = inference_job.collect_checkpoint_paths(roots, 5)

            self.assertEqual(len(paths), 5)
            self.assertEqual(paths[3].name, "best_metric_model.pth")
            self.assertIn("fold_3", str(paths[3]))

    def test_select_inference_cases_uses_reserved_test_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            split_csv = Path(tmp) / "cv_splits_qc.csv"
            split_csv.write_text(
                "patient_id,fold\n"
                "TAVI_TRAIN,0\n"
                "TAVI_TEST_A,-1\n"
                "TAVI_TEST_B,-1\n"
            )
            for patient_id in ("TAVI_TRAIN", "TAVI_TEST_A", "TAVI_TEST_B"):
                patient_dir = root / patient_id
                patient_dir.mkdir(parents=True)
                (patient_dir / "CT_LATE.nii.gz").write_bytes(b"image")
                (patient_dir / "registration_mask.nii.gz").write_bytes(b"label")

            cases = inference_job.select_cases_from_split(root, split_csv, test_fold=-1)

            self.assertEqual(
                [Path(case["image"]).parent.name for case in cases],
                ["TAVI_TEST_A", "TAVI_TEST_B"],
            )


if __name__ == "__main__":
    unittest.main()
