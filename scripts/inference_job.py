"""Standalone ensemble inference entrypoint for Azure ML jobs."""

import argparse
import random
import re
from contextlib import nullcontext
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset, MetaTensor, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import (
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


DEFAULT_ROI_SIZE = (96, 96, 96)
DEFAULT_TARGET_SPACING = (0.5, 0.5, 0.5)
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


def collect_checkpoint_paths(checkpoints_root, fold_count):
    checkpoints_root = Path(checkpoints_root)
    checkpoint_paths = []
    missing_paths = []

    for fold_idx in range(fold_count):
        candidate_paths = [
            checkpoints_root
            / f"fold{fold_idx}"
            / "named-outputs"
            / "output_model"
            / f"fold_{fold_idx}"
            / "best_metric_model.pth",
            checkpoints_root / f"fold_{fold_idx}" / "best_metric_model.pth",
            checkpoints_root
            / f"fold_{fold_idx}"
            / "named-outputs"
            / "output_model"
            / "best_metric_model.pth",
            checkpoints_root
            / f"fold_{fold_idx}"
            / "named-outputs"
            / "output_model"
            / f"fold_{fold_idx}"
            / "latest_checkpoint.pth",
        ]

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
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            CropForegroundd(keys=["image"], source_key="image"),
            Spacingd(keys=["image"], pixdim=target_spacing, mode=("bilinear",)),
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
            ),
        ]
    )

    return infer_transforms, post_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble inference for myocardium segmentation")
    parser.add_argument("--input_data", type=str, required=True, help="Root folder with TAVI_* cases")
    parser.add_argument(
        "--checkpoints_root",
        type=str,
        required=True,
        help="Root folder containing fold checkpoint outputs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the predicted mask will be written",
    )
    parser.add_argument("--fold_count", type=int, default=5, help="Number of folds to ensemble")
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
        default=DEFAULT_ROI_SIZE,
        help="Sliding window ROI size",
    )
    parser.add_argument(
        "--target_spacing",
        type=float,
        nargs=3,
        default=DEFAULT_TARGET_SPACING,
        help="Voxel spacing used during preprocessing",
    )
    parser.add_argument(
        "--sw_batch_size",
        type=int,
        default=4,
        help="Sliding window batch size",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.8,
        help="Sliding window overlap",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_data = Path(args.input_data)
    checkpoints_root = Path(args.checkpoints_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_cases = find_valid_cases(input_data)
    test_case = select_inference_case(
        valid_cases,
        seed=args.seed,
        unseen_offset=args.unseen_offset,
        case_name=args.case_name,
    )

    checkpoint_paths = collect_checkpoint_paths(checkpoints_root, args.fold_count)

    print("Ensembling these checkpoints:", flush=True)
    for path in checkpoint_paths:
        print(path, flush=True)

    print(f"Running inference on: {test_case['image']}", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_context = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()

    infer_transforms, post_transforms = build_transforms(args.target_spacing, output_dir)
    dataset = Dataset(data=[test_case], transform=infer_transforms)
    loader = DataLoader(dataset, batch_size=1)

    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
    ).to(device)

    with torch.inference_mode():
        for batch in loader:
            inputs = batch["image"].to(device)
            ensemble_probs = None

            for checkpoint_path in checkpoint_paths:
                load_checkpoint_into_model(model, checkpoint_path, device)

                with amp_context:
                    logits = sliding_window_inference(
                        inputs,
                        tuple(args.roi_size),
                        args.sw_batch_size,
                        model,
                        overlap=args.overlap,
                    )
                    probs = torch.softmax(logits, dim=1)

                ensemble_probs = probs if ensemble_probs is None else ensemble_probs + probs

            ensemble_probs = ensemble_probs / len(checkpoint_paths)

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

    print(f"Saved ensemble mask under: {output_dir}", flush=True)


if __name__ == "__main__":
    main()