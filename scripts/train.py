import os
import re
import random
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from azureml.core import Workspace, Dataset as AzureDataset

import torch
from monai.utils import set_determinism
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import Dataset, PersistentDataset, DataLoader, decollate_batch

from monai.transforms import (
    AsDiscrete, EnsureChannelFirstd, Compose,
    CropForegroundd, LoadImaged, Orientationd, RandFlipd,
    RandCropByPosNegLabeld, ScaleIntensityRanged, Spacingd,
    RandRotate90d, ResizeWithPadOrCropd
)
from monai.networks.nets import SegResNet

# ---- 1. DYNAMIC FILE LOADING ----
def get_files(input_dir, train_size=0.9, seed=1303):
    input_dir = Path(input_dir)
    valid_files = []
    pattern = re.compile(r"^TAVI_\d+$")
    
    for patient_folder in input_dir.iterdir():
        if patient_folder.is_dir() and pattern.match(patient_folder.name):
            img_path = patient_folder / "CT_ANGIO.nii.gz"
            lbl_path = patient_folder / "TotalSegmentator" / "CT_ANGIO" / "heartchambers_highres" / "heart_myocardium.nii.gz"
            
            if img_path.exists() and lbl_path.exists():
                valid_files.append({"image": str(img_path), "label": str(lbl_path)})
                
    random.seed(seed)
    random.shuffle(valid_files)
    valid_files = valid_files[:200]
    
    split_idx = int(len(valid_files) * train_size)
    return valid_files[:split_idx], valid_files[split_idx:], valid_files

# ---- 2. TRANSFORMS ----
ROI_SIZE = (96, 96, 96)
TARGET_SPACING = (0.5, 0.5, 0.5)

def get_transforms():
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CropForegroundd(keys=["image", "label"], source_key="image"), # MOVED UP!
        Spacingd(keys=["image", "label"], pixdim=TARGET_SPACING, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=ROI_SIZE, pos=3, neg=1, num_samples=4, image_key="image", image_threshold=0, allow_smaller=True),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=ROI_SIZE, mode='constant'),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3)
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Spacingd(keys=["image", "label"], pixdim=TARGET_SPACING, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    ])
    
    return train_transforms, val_transforms


# ---- 3. MAIN TRAINING FUNCTION ----
def main(args):
    set_determinism(1303)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- START STANDALONE MOUNT ---
    print("Connecting to Azure Workspace and mounting data...")
    ws = Workspace(
        subscription_id="ab211f7b-463f-4833-9605-d260e596a35a",
        resource_group="73da10b4-5dff-54e2-db0d-3a1fab882485",
        workspace_name="73da10b45dff54e2db0d3a1fab882485"
    )
    datastore = ws.datastores["73da10b45dff54e2db0d3a1fab882485"]
    nifti_dataset = AzureDataset.File.from_files(path=(datastore, 'NIFTI/'))
    
    mount_context = nifti_dataset.mount()
    mount_context.start()
    
    try:
        input_dfs_dir = Path(mount_context.mount_point)
        print(f"Data successfully mounted at: {input_dfs_dir}")
        
        train_files, val_files, all_files = get_files(input_dfs_dir, train_size=0.9, seed=1303)
        print(f"Found {len(all_files)} valid cases. Train: {len(train_files)} | Val: {len(val_files)}")

        train_transforms, val_transforms = get_transforms()

        # Create a fast local folder on the VM to store the pre-processed tensors
        cache_dir = Path("./monai_cache") # /mnt/ is usually the fast, temporary NVMe drive on Azure VMs
        cache_dir.mkdir(parents=True, exist_ok=True)

        # train_ds = Dataset(data=train_files, transform=train_transforms)
        # val_ds = Dataset(data=val_files, transform=val_transforms)
        train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=cache_dir)
        val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=cache_dir)

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            init_filters=16,         
            blocks_down=[1, 2, 2, 4], 
            blocks_up=[1, 1, 1],
        ).to(device)

        loss_function = DiceFocalLoss(include_background=False, to_onehot_y=True, softmax=True, gamma=2.0, lambda_dice=0.5, lambda_focal=0.5, alpha=0.75)
        torch.backends.cudnn.benchmark = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

        post_label = AsDiscrete(to_onehot=2)
        post_pred = AsDiscrete(argmax=True, to_onehot=2)
        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

        # AMP Scaler Initialization
        scaler = torch.amp.GradScaler('cuda')

        # Training Config
        max_iterations = 5000 
        eval_num = 500
        accumulation_steps = 4  # batch_size=2 * 4 = effective batch of 8

        global_step = 0
        dice_val_best = 0.0

        optimizer.zero_grad()

        while global_step < max_iterations:
            model.train()
            epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps)", dynamic_ncols=True)
            
            for step_idx, batch in enumerate(epoch_iterator):
                x, y = (batch["image"].to(device), batch["label"].to(device))
                
                # 1. AMP Forward Pass
                with torch.amp.autocast('cuda'):
                    logit_map = model(x)
                    loss = loss_function(logit_map, y)
                    loss = loss / accumulation_steps
                
                # 2. AMP Backward Pass (scaled)
                scaler.scale(loss).backward()
                
                # 3. Step Optimizer and Update Scaler
                if ((step_idx + 1) % accumulation_steps == 0) or (step_idx + 1 == len(train_loader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                display_loss = loss.item() * accumulation_steps
                epoch_iterator.set_description(f"Training ({global_step} / {max_iterations} Steps) (loss={display_loss:2.5f})")
                
                # ---- VALIDATION BLOCK ----
                if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
                    model.eval()
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_inputs, val_labels = (val_batch["image"].to(device), val_batch["label"].to(device))
                            
                            # Apply AMP autocast to validation inference as well to save memory
                            with torch.cuda.amp.autocast():
                                val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, 4, model, overlap=0.8)
                            
                            val_labels_list = decollate_batch(val_labels)
                            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                            
                            val_outputs_list = decollate_batch(val_outputs)
                            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                            
                            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                        
                        dice_val = dice_metric.aggregate().item()
                        dice_metric.reset()
                        
                        if dice_val > dice_val_best:
                            dice_val_best = dice_val
                            torch.save(model.state_dict(), os.path.join(out_dir, "best_metric_model.pth"))
                            print(f"\nModel Saved! New Best Dice: {dice_val_best:.4f}")
                        else:
                            print(f"\nValidation complete. Current Dice: {dice_val:.4f} (Best: {dice_val_best:.4f})")
                    
                    model.train()
                    
                global_step += 1
                if global_step >= max_iterations:
                    break
    finally:
        # This ensures the VM cleans up the FUSE mount when the script finishes or crashes!
        print("\nUnmounting Azure datastore...")
        mount_context.stop()
        print("Cleanup complete.")        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Myocardium Segmentation Training")
    parser.add_argument("--out_dir", type=str, default="./output/models", help="Directory to save the trained model")
    
    args = parser.parse_args()
    main(args)