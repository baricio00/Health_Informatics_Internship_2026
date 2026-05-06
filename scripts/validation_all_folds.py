import os, sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import nibabel as nib
from omegaconf import OmegaConf

from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import AsDiscrete, SaveImage

from lems_ct.src.models.model import get_segresnet
from lems_ct.src.utils.transforms import get_transforms, get_post_transforms
from lems_ct.src.utils.data import get_files_from_csv
from lems_ct.src.metrics.utils import calculate_dice_split

def run_fold_inference(fold_idx, weight_dir, args, cfg, device, master_results):
    print(f"\n--- [FOLD {fold_idx}] Initialization ---")
    
    # 1. Setup Fold Output Paths
    fold_output_dir = Path(args.output_dir) / f"fold_{fold_idx}"
    mask_dir = fold_output_dir / "segmentations"
    mask_dir.mkdir(parents=True, exist_ok=True)

    # 2. Get the EXACT same val_transforms used in training
    # We ignore train_transforms here and only take the second returned value
    _, val_transforms = get_transforms(**cfg.transforms)
    post_transforms = get_post_transforms(val_transforms, mask_dir)

    # 3. Load Data
    _, val_files = get_files_from_csv(args.input_data, args.split_csv, fold_idx)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    # 4. Load Model
    model = get_segresnet(**cfg.model).to(device)
    
    # Path logic for mounted Azure outputs
    checkpoint_path = Path(weight_dir) / f"fold_{fold_idx}" / "best_metric_model.pth"
    print(f"Loading weights: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # 5. Setup Post-processing
    # Note: Using out_channels from config for multi-class support
    post_label = AsDiscrete(to_onehot=cfg.model.out_channels)
    post_pred_metric = AsDiscrete(argmax=True, to_onehot=cfg.model.out_channels)

    print(f"Running inference on {len(val_loader)} volumes...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            patient_id = val_files[i]["image"].split('/')[-2]

            with torch.amp.autocast("cuda"):
                outputs = sliding_window_inference(
                    inputs,
                    cfg.transforms.roi_size,
                    cfg.inference.sw_batch_size,
                    model,
                    overlap=cfg.inference.overlap,
                    mode=cfg.inference.mode,
                )
                
                # sliding window inference returns a batched tensor of shape (B, C, H, W, D)
                # outputs contains raw logits in the "transformed" space due to the application
                # of validation transforms. One channel per class: in this case, 2 (bg and fg).
                # Metrics must be computed in this transformed space; However, transforms must 
                # be inverted before saving the masks to disk.
            
            # --- Decollate and apply  ---
            # what decollate_batch does is take a batched tensor of shape (B, ...) and splits it
            # into a list of B tensors of shape (...). With batch_size=1 as in our case
            # the batched tensor has B=1, hence decollate_batch will produce a one-element list.
            # still fundamental because it gets rid of the batch axis.
            val_labels_oh = [post_label(x) for x in decollate_batch(labels)]
            val_output_oh = [post_pred_metric(x) for x in decollate_batch(outputs)]
            
            # remember that post_label has to_onehot=2
            # Dice for multi-class requires one-hot tensors, so both labels and predictions
            # need to be one-hot encoded.
            # after decollating labels, we have a one-element list [(1, H, W, D)] # because in_channels is 1
            # one-hot encoding means having (1, H, W, D) for both 0 and 1

            
            # Now for the predicitons; remember that post_pred_metric has argmax=True and to_onehot=2
            # after decollating preds, we have a one-element list [(C, H, W, D)] # out_channels is 2
            # it does two things simultaneously: argmax collapses the 2-channel raw logits down to
            # a single-channel class-index map (basically assigning each pixel to either 0 or 1 i.e. the class index)
            # depending on the magnitude of the probability assigned by the model to each class
            # then the to_onehot re-expands that into C (in our case, 2) binary tensors.
            
            # after these ops, both val_labels_ouh and val_output_oh have the same shape and semantics.
            # since batch_size=1, both of these lists have a single element which is a tensor (C, H, W, D)
            # with C = 2 in our case, background and foreground


            # Metric Calculation
            dice_scores, _, _ = calculate_dice_split(
                val_output_oh[0].flatten(1), # flattens dims 1 onward -> (C, H*W*D)
                val_labels_oh[0].flatten(1), 
                cfg.model.out_channels
            ) # dice_scores is a length-C vector containing Dice scores per class
            dice_numpy = dice_scores.cpu().numpy()
            # dice_numpy[0] background Dice, dice_numpy[1] foreground Dice

            # Filter for foreground (indices 1+)
            case_stat = {
                "fold": fold_idx,
                "patient_id": patient_id
            }
            for c in range(1, cfg.model.out_channels):
                case_stat[f"dice_class_{c}"] = dice_numpy[c]
            
            if cfg.model.out_channels > 2:
                case_stat["mean_dice"] = np.nanmean(dice_numpy[1:])
    
            master_results.append(case_stat)


            # Save Segmentation Mask
            batch["pred"] = outputs
            # we add the sliding window output to the batch (keys: image, label and now pred)
            for d in decollate_batch(batch): # outputs a list of B tensors of shape (C, H, W, D)
                post_transforms(d) # these post transforms will target "pred" and "image" keys explicitely
                
            dice_str = " | ".join(
                f"c{c}: {dice_numpy[c]:.4f}" for c in range(1, cfg.model.out_channels)
            )
            print(f"  [{fold_idx}] {patient_id} | {dice_str}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--split_csv", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--w0", type=str, required=True)
    parser.add_argument("--w1", type=str, required=True)
    parser.add_argument("--w2", type=str, required=True)
    parser.add_argument("--w3", type=str, required=True)
    parser.add_argument("--w4", type=str, required=True)
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master_results = []

    weight_map = {0: args.w0, 1: args.w1, 2: args.w2, 3: args.w3, 4: args.w4}

    # Sequential Loop
    for fold_idx in range(5):
        run_fold_inference(fold_idx, weight_map[fold_idx], args, cfg, device, master_results)

    # Consolidated CSV
    df = pd.DataFrame(master_results)
    df.to_csv(Path(args.output_dir) / "all_folds_validation_report.csv", index=False)
    print("\n✅ CV Inference Complete. Report saved.")

if __name__ == "__main__":
    main()