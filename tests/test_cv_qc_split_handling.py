import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lems_ct.src.utils import data


def write_fake_nifti_gz(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x1f\x8bnot-a-real-nifti")


class CVQCSplitHandlingTests(unittest.TestCase):
    def test_fold_minus_one_is_reserved_for_test_not_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            split_csv = Path(tmp) / "cv_splits_qc.csv"
            split_csv.write_text(
                "patient_id,fold\n"
                "TAVI_TRAIN,1\n"
                "TAVI_VAL,0\n"
                "TAVI_TEST,-1\n"
            )

            for patient_id in ("TAVI_TRAIN", "TAVI_VAL", "TAVI_TEST"):
                write_fake_nifti_gz(root / patient_id / "CT_LATE.nii.gz")
                write_fake_nifti_gz(root / patient_id / "registration_mask.nii.gz")

            with mock.patch.object(data, "PROJECT_ROOT", Path(tmp)):
                train_files, val_files = data.get_files_from_csv(root, split_csv, 0)

            train_patients = {Path(item["image"]).parent.name for item in train_files}
            val_patients = {Path(item["image"]).parent.name for item in val_files}

            self.assertEqual(train_patients, {"TAVI_TRAIN"})
            self.assertEqual(val_patients, {"TAVI_VAL"})
            self.assertNotIn("TAVI_TEST", train_patients | val_patients)


if __name__ == "__main__":
    unittest.main()
