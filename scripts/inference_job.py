"""Standalone ensemble inference entrypoint for Azure ML jobs."""
from __future__ import annotations

import argparse
import csv
import random
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset, MetaTensor, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
)
from omegaconf import OmegaConf

from lems_ct.src.metrics.utils import calculate_dice_split
from lems_ct.src.models.model import get_segresnet
from lems_ct.src.utils.data import resolve_data_root

DEFAULT_ROI_SIZE = (96, 96, 96)
DEFAULT_TARGET_SPACING = (1.0, 1.0, 1.0)
PATIENT_PATTERN = re.compile(r"^TAVI_\d+$")


def find_valid_cases(input_dir):
    valid_cases = []

    for patient_folder in sorted(Path(input_dir).iterdir()):
        if not patient_folder.is_dir() or not PATIENT_PATTERN.match(patient_folder.name):
            continue

        image_path = patient_folder / "CT_LATE.nii.gz"
        label_path = patient_folder / "registration_mask.nii.gz"

        if image_path.exists() and label_path.exists():
            valid_cases.append({"image": str(image_path), "label": str(label_path)})

    return valid_cases


def select_inference_case(valid_cases, seed, unseen_offset, case_name=None):
    if not valid_cases:
        raise ValueError("No valid cases were found in the input data directory.")

    if case_name:
        for case in valid_cases:
            if Path(case["image"]).parent.name == case_name:
                return case
        raise ValueError(f"Could not find case '{case_name}' in the available data.")

    shuffled_cases = list(valid_cases)
    random.seed(seed)
    random.shuffle(shuffled_cases)

    if len(shuffled_cases) > unseen_offset:
        return shuffled_cases[unseen_offset]

    return shuffled_cases[-1]


def _checkpoint_candidates(root, fold_idx):
    root = Path(root)
    return [
        root / f"fold_{fold_idx}" / "best_metric_model.pth",
        root / f"fold_{fold_idx}" / "latest_checkpoint.pth",
        root / "output_model" / f"fold_{fold_idx}" / "best_metric_model.pth",
        root / "output_model" / f"fold_{fold_idx}" / "latest_checkpoint.pth",
        root
        / f"fold{fold_idx}"
        / "named-outputs"
        / "output_model"
        / f"fold_{fold_idx}"
        / "best_metric_model.pth",
        root
        / f"fold_{fold_idx}"
        / "named-outputs"
        / "output_model"
        / f"fold_{fold_idx}"
        / "best_metric_model.pth",
        root
        / f"fold_{fold_idx}"
        / "named-outputs"
        / "output_model"
        / f"fold_{fold_idx}"
        / "latest_checkpoint.pth",
    ]


def collect_checkpoint_paths(checkpoints_root, fold_count):
    if isinstance(checkpoints_root, (list, tuple)):
        if len(checkpoints_root) != fold_count:
            raise ValueError(
                f"Expected {fold_count} checkpoint roots, received {len(checkpoints_root)}."
            )
        roots_by_fold = [Path(root) for root in checkpoints_root]
    else:
        checkpoints_root = Path(checkpoints_root)
        roots_by_fold = [checkpoints_root for _ in range(fold_count)]

    checkpoint_paths = []
    missing_paths = []

    for fold_idx in range(fold_count):
        candidate_paths = _checkpoint_candidates(roots_by_fold[fold_idx], fold_idx)

        checkpoint_path = next((path for path in candidate_paths if path.exists()), None)
        if checkpoint_path is None:
            missing_paths.append(candidate_paths[0])
            continue

        checkpoint_paths.append(checkpoint_path)

    if missing_paths:
        raise FileNotFoundError(
            "Missing checkpoints:\n" + "\n".join(str(path) for path in missing_paths)
        )

    return checkpoint_paths


def select_cases_from_split(input_dir, split_csv, test_fold=-1, case_name=None):
    input_dir = resolve_data_root(input_dir)
    df = pd.read_csv(split_csv)
    test_rows = df[df["fold"].astype(int) == int(test_fold)]

    cases = []
    for patient_id in test_rows["patient_id"].astype(str):
        if case_name and patient_id != case_name:
            continue

        image_path = input_dir / patient_id / "CT_LATE.nii.gz"
        label_path = input_dir / patient_id / "registration_mask.nii.gz"
        if image_path.exists() and label_path.exists():
            cases.append({"image": str(image_path), "label": str(label_path)})

    if case_name and not cases:
        raise ValueError(
            f"Could not find case '{case_name}' in split fold {test_fold} under {input_dir}."
        )
    if not cases:
        raise ValueError(
            f"No valid cases were found for split fold {test_fold} using {split_csv}."
        )

    return cases


def load_checkpoint_into_model(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format in {checkpoint_path}")

    cleaned_state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }

    model.load_state_dict(cleaned_state_dict)
    model.eval()


def build_transforms(target_spacing, output_dir):
    infer_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Spacingd(
                keys=["image", "label"],
                pixdim=target_spacing,
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-100.0,
                a_max=400.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
    )

    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=infer_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True),
            SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=str(output_dir),
                output_postfix="pred_mask",
                output_ext=".nii.gz",
                resample=False,
                writer="ITKWriter",
                output_dtype=np.uint8,
            ),
        ]
    )

    return infer_transforms, post_transforms


def checkpoint_roots_from_args(args):
    fold_roots = [getattr(args, f"w{fold_idx}") for fold_idx in range(args.fold_count)]
    provided_roots = [root for root in fold_roots if root]
    if provided_roots:
        if len(provided_roots) != args.fold_count:
            raise ValueError(
                f"Provide all --w0..--w{args.fold_count - 1} checkpoint roots, "
                "or use --checkpoints_root."
            )
        return fold_roots

    if not args.checkpoints_root:
        raise ValueError("Provide --checkpoints_root or all per-fold --w0..--w4 inputs.")
    return args.checkpoints_root


def cases_from_args(args, input_data):
    if args.split_csv:
        return select_cases_from_split(
            input_data,
            args.split_csv,
            test_fold=args.test_fold,
            case_name=args.case_name or None,
        )

    valid_cases = find_valid_cases(input_data)
    return [
        select_inference_case(
            valid_cases,
            seed=args.seed,
            unseen_offset=args.unseen_offset,
            case_name=args.case_name or None,
        )
    ]


def one_hot_from_label_map(label_map, out_channels):
    one_hot = torch.nn.functional.one_hot(
        label_map.long(), num_classes=out_channels
    )
    return one_hot.movedim(-1, 1).float()


def foreground_dice(pred_onehot, labels, out_channels):
    post_label = AsDiscrete(to_onehot=out_channels)
    labels_onehot = [post_label(item) for item in decollate_batch(labels)]
    pred_onehot_items = decollate_batch(pred_onehot)
    dice_scores, _, _ = calculate_dice_split(
        pred_onehot_items[0].flatten(1),
        labels_onehot[0].flatten(1),
        out_channels,
    )
    dice_numpy = dice_scores.detach().cpu().numpy()
    return {
        f"dice_class_{class_idx}": float(dice_numpy[class_idx])
        for class_idx in range(1, out_channels)
    }


def write_metrics_csv(output_dir, rows):
    if not rows:
        return

    output_path = Path(output_dir) / "ensemble_test_metrics.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved ensemble test metrics: {output_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble inference for myocardium segmentation")
    parser.add_argument("--input_data", type=str, required=True, help="Root folder with TAVI_* cases")
    parser.add_argument(
        "--checkpoints_root",
        type=str,
        default=None,
        help="Root folder containing fold checkpoint outputs",
    )
    for fold_idx in range(5):
        parser.add_argument(
            f"--w{fold_idx}",
            type=str,
            default=None,
            help=f"Mounted output_model folder for fold {fold_idx}",
        )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the predicted mask will be written",
    )
    parser.add_argument("--fold_count", type=int, default=5, help="Number of folds to ensemble")
    parser.add_argument(
        "--split_csv",
        type=str,
        default=None,
        help="CSV containing patient_id and fold. Fold -1 is used as the test set by default.",
    )
    parser.add_argument("--test_fold", type=int, default=-1, help="Fold label reserved for testing")
    parser.add_argument("--config", type=str, default="config/train_config.yaml")
    parser.add_argument(
        "--case_name",
        type=str,
        default=None,
        help="Optional patient folder name to infer on instead of selecting an unseen case",
    )
    parser.add_argument("--seed", type=int, default=1303, help="Random seed used for case selection")
    parser.add_argument(
        "--unseen_offset",
        type=int,
        default=200,
        help="Index offset used by the notebook cell to select an unseen sample",
    )
    parser.add_argument(
        "--roi_size",
        type=int,
        nargs=3,
        default=None,
        help="Sliding window ROI size",
    )
    parser.add_argument(
        "--target_spacing",
        type=float,
        nargs=3,
        default=None,
        help="Voxel spacing used during preprocessing",
    )
    parser.add_argument(
        "--sw_batch_size",
        type=int,
        default=None,
        help="Sliding window batch size",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="Sliding window overlap",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_data = Path(args.input_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)

    test_cases = cases_from_args(args, input_data)

    checkpoint_paths = collect_checkpoint_paths(checkpoint_roots_from_args(args), args.fold_count)

    print("Ensembling these checkpoints:", flush=True)
    for path in checkpoint_paths:
        print(path, flush=True)

    print(f"Running ensemble inference on {len(test_cases)} case(s).", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_context = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
    roi_size = tuple(args.roi_size or cfg.transforms.get("roi_size", DEFAULT_ROI_SIZE))
    target_spacing = tuple(
        args.target_spacing or cfg.transforms.get("target_spacing", DEFAULT_TARGET_SPACING)
    )
    sw_batch_size = int(args.sw_batch_size or cfg.inference.get("sw_batch_size", 4))
    overlap = float(args.overlap if args.overlap is not None else cfg.inference.get("overlap", 0.5))
    mode = cfg.inference.get("mode", "gaussian")
    out_channels = int(cfg.model.get("out_channels", 2))

    infer_transforms, post_transforms = build_transforms(target_spacing, output_dir)
    dataset = Dataset(data=test_cases, transform=infer_transforms)
    loader = DataLoader(dataset, batch_size=1)

    model = get_segresnet(**cfg.model).to(device)
    metric_rows = []
    from scripts.lcc_postprocessing import discretize_clean_ensemble_probs

    with torch.inference_mode():
        for case_idx, batch in enumerate(loader):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            ensemble_probs = None

            for checkpoint_path in checkpoint_paths:
                load_checkpoint_into_model(model, checkpoint_path, device)

                with amp_context:
                    logits = sliding_window_inference(
                        inputs,
                        roi_size,
                        sw_batch_size,
                        model,
                        overlap=overlap,
                        mode=mode,
                    )
                    probs = torch.softmax(logits, dim=1)

                ensemble_probs = probs if ensemble_probs is None else ensemble_probs + probs

            ensemble_probs = ensemble_probs / len(checkpoint_paths)
            raw_onehot = one_hot_from_label_map(torch.argmax(ensemble_probs, dim=1), out_channels)
            lcc_label_map = discretize_clean_ensemble_probs(ensemble_probs)
            lcc_onehot = one_hot_from_label_map(lcc_label_map, out_channels)

            src_meta = batch["image"].meta if hasattr(batch["image"], "meta") else batch["image_meta_dict"]
            predictions = MetaTensor(ensemble_probs, meta=src_meta)

            batch_data = {
                "image": batch["image"],
                "pred": predictions,
            }

            for item in decollate_batch(batch_data):
                if "image_meta_dict" not in item:
                    item["image_meta_dict"] = item["image"].meta if hasattr(item["image"], "meta") else src_meta
                if "pred_meta_dict" not in item:
                    item["pred_meta_dict"] = dict(item["image_meta_dict"])

                post_transforms(item)

            patient_id = Path(test_cases[case_idx]["image"]).parent.name
            row = {"patient_id": patient_id}
            row.update({f"raw_{key}": value for key, value in foreground_dice(raw_onehot, labels, out_channels).items()})
            row.update({f"lcc_{key}": value for key, value in foreground_dice(lcc_onehot, labels, out_channels).items()})
            metric_rows.append(row)
            print(f"  {patient_id} | {row}", flush=True)

    write_metrics_csv(output_dir, metric_rows)
    print(f"Saved ensemble masks under: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
