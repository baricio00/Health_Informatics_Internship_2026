import csv
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_local_path(path):
    path = Path(path).expanduser()
    if (
        path.is_absolute()
        and len(path.parts) > 1
        and path.parts[1] == "notebooks"
        and not Path("/notebooks").exists()
    ):
        return PROJECT_ROOT / "notebooks" / Path(*path.parts[2:])
    return path


def resolve_data_root(input_dir):
    input_dir = resolve_local_path(input_dir)
    azure_layout = input_dir / "DATA_REGISTERED_FIXED_LATE"
    return azure_layout if azure_layout.exists() else input_dir


def is_valid_gzip_file(path):
    try:
        with Path(path).open("rb") as file_obj:
            return file_obj.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def is_valid_nifti_path(path):
    path = Path(path)
    if not path.exists():
        return False, "missing"
    if path.suffix == ".gz" and not is_valid_gzip_file(path):
        return False, "not_gzip"
    return True, ""


def write_skipped_files_report(skipped_files, fold_idx):
    if not skipped_files:
        return

    report_path = PROJECT_ROOT / "data" / f"skipped_invalid_files_fold_{fold_idx}.csv"
    with report_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["patient_id", "fold", "reason", "image", "label"],
        )
        writer.writeheader()
        writer.writerows(skipped_files)
    print(f"Skipped {len(skipped_files)} invalid/missing scans. Report: {report_path}")


def get_files_from_csv(input_dir, csv_path, fold_idx):
    input_dir = resolve_data_root(input_dir)
    print(f"Using data root: {input_dir}")
    print(f"Using split CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    train_files = []
    val_files = []
    skipped_files = []
    
    for _, row in df.iterrows():
        patient_id = str(row["patient_id"])
        current_fold = int(row["fold"])
        
        # Dynamically build the absolute path using the Azure mount directory
        img_path = input_dir / patient_id / "CT_LATE.nii.gz"
        lbl_path = input_dir / patient_id / "registration_mask.nii.gz"
        
        image_ok, image_reason = is_valid_nifti_path(img_path)
        label_ok, label_reason = is_valid_nifti_path(lbl_path)
        if not image_ok or not label_ok:
            reasons = []
            if not image_ok:
                reasons.append(f"image_{image_reason}")
            if not label_ok:
                reasons.append(f"label_{label_reason}")
            skipped_files.append(
                {
                    "patient_id": patient_id,
                    "fold": current_fold,
                    "reason": ";".join(reasons),
                    "image": str(img_path),
                    "label": str(lbl_path),
                }
            )
            continue

        data_dict = {"image": str(img_path), "label": str(lbl_path)}

        # Route to Train or Val based on the fold argument
        if current_fold == fold_idx:
            val_files.append(data_dict)
        else:
            train_files.append(data_dict)
                
    write_skipped_files_report(skipped_files, fold_idx)
    print(f"Loaded {len(train_files)} training scans and {len(val_files)} validation scans for Fold {fold_idx}.")
    return train_files, val_files
