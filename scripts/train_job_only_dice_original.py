import os
import time
import argparse
import mlflow
from omegaconf import OmegaConf
from pathlib import Path
from datetime import timedelta
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from monai.utils import set_determinism
from monai.losses import DiceFocalLoss

# from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.data import PersistentDataset, DataLoader, decollate_batch
from monai.transforms import AsDiscrete

from lems_ct.src.models.model import get_segresnet
from lems_ct.src.utils.transforms import get_transforms
from lems_ct.src.utils.data import get_files_from_csv
from lems_ct.src.utils.misc import (
    update_ema_variables,
    exp_lr_scheduler_with_warmup,
    concat_all_gather,
    train_collate_fn
)
from lems_ct.src.metrics.utils import calculate_dice_split

# ==========================================
# Core Logic
# ==========================================

def validation_ddp(
    model, val_loader, loss_function, post_label, post_pred, device, cfg, args
):
    """
    Handles the validation loop across multiple GPUs.
    """
    model.eval() # Set model to evaluation mode (disables dropout, changes batch norm behavior)

    dice_list = []
    unique_labels_list = []

    local_val_loss = 0.0
    local_steps = 0

    # The rank identifies which GPU is running this specific process (0 is the master/main GPU)
    global_rank = int(os.environ.get("RANK", 0))

    if global_rank == 0:
        print("\nEvaluating...", flush=True)

    # Disable gradient calculation for validation to save memory and compute
    with torch.no_grad():
        for step, val_batch in enumerate(val_loader, start=1):
            if global_rank == 0:
                print(f"\nValidating volume {step}/{len(val_loader)}...", flush=True)

            # Synchronize CUDA before measuring time to ensure accurate benchmarking
            torch.cuda.synchronize()
            t0 = time.time()
                
            val_inputs = val_batch["image"].to(device).float()
            val_labels = val_batch["label"].to(device).long()

            # Use Automatic Mixed Precision (AMP) for faster inference
            with torch.amp.autocast("cuda"):
                # sliding_window_inference is crucial for medical imaging. 3D volumes 
                # are often too large to fit in GPU memory at once. This function breaks 
                # the volume into overlapping patches, infers on each, and stitches them back together.
                val_outputs = sliding_window_inference(
                    val_inputs,
                    cfg.transforms.roi_size,
                    cfg.inference.sw_batch_size,
                    model.module, # .module is required because the model is wrapped in DDP
                    overlap=cfg.inference.overlap,
                    mode=cfg.inference.mode,
                )
                local_val_loss += loss_function(val_outputs, val_labels).item()
                local_steps += 1

            torch.cuda.synchronize()
            t1 = time.time()
            if global_rank == 0:
                print(f"  -> Inference: {t1-t0:.2f}s", flush=True)

            # --- Data Post-processing ---
            # Decollate splits the batch dimension back into a list of individual tensors
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]

            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]

            label_pred = val_output_convert[0].cpu()
            labels = val_labels_convert[0].cpu().bool()

            # Extract unique classes present in this specific ground truth volume
            unique_labels = torch.unique(labels).cpu().numpy()
            # Pad the unique labels array to a fixed size so it can be safely gathered across GPUs
            unique_labels = np.pad(
                unique_labels,
                (100 - len(unique_labels), 0),
                "constant",
                constant_values=0,
            )

            t2 = time.time()
            if global_rank == 0:
                print(f"  -> Discretize & CPU Move: {t2-t1:.2f}s", flush=True)

            t3 = time.time()

            # Calculate the Dice similarity coefficient
            tmp_dice_list, _, _ = calculate_dice_split(
                label_pred.view(-1, 1), labels.view(-1, 1), cfg.model.out_channels
            )

            # Format tensors to include a batch dimension so they can be concatenated during the all_gather
            tmp_dice_list = tmp_dice_list.unsqueeze(0)
            unique_labels = np.expand_dims(unique_labels, axis=0)

            t4 = time.time()
            if global_rank == 0:
                print(f"  -> Dice Score: {t4-t3:.2f}s", flush=True)

            # --- Distributed Gathering ---
            # Collect metrics from all active GPUs
            tmp_dice_list = concat_all_gather(tmp_dice_list.cuda(device))
            unique_labels = (
                concat_all_gather(torch.from_numpy(unique_labels).cuda(device))
                .cpu()
                .numpy()
            )

            t5 = time.time()
            if global_rank == 0:
                print(f"  -> DDP Gather: {t5-t4:.2f}s", flush=True)

            # Exclude background (index 0) from the dice list to focus on foreground structures
            tmp_dice_list = tmp_dice_list.cpu().numpy()[:, 1:]

            for idx in range(len(tmp_dice_list)):
                dice_list.append(tmp_dice_list[idx])
                unique_labels_list.append(unique_labels[idx])

    # 1. Average local validation loss across all GPUs
    local_loss_avg = torch.tensor([local_val_loss / max(1, local_steps)], device=device)
    dist.all_reduce(local_loss_avg, op=dist.ReduceOp.SUM) # Sums the loss across GPUs
    global_val_loss = (local_loss_avg / dist.get_world_size()).item() # Divides by number of GPUs

    # 2. Remove padded duplicate samples from DistributedSampler
    # DistributedSampler pads the dataset to ensure every GPU gets the exact same number of batches.
    # We must remove these duplicate "dummy" samples before calculating our final metrics.
    world_size = dist.get_world_size()
    dataset_len = len(val_loader.dataset)
    padding_size = (
        0
        if (dataset_len % world_size) == 0
        else world_size - (dataset_len % world_size)
    )

    for _ in range(padding_size):
        if len(dice_list) > 0:
            dice_list.pop()
            unique_labels_list.pop()

    # 3. Filter metrics to only include classes actually present in the Ground Truth
    # If a class doesn't exist in a specific patient scan, predicting 0 is correct, 
    # but the standard Dice math breaks down (divide by zero). We filter those out here.
    out_dice = [[] for _ in range(cfg.model.out_channels - 1)]

    for idx in range(len(dice_list)):
        for cls in range(0, cfg.model.out_channels - 1):
            if cls + 1 in unique_labels_list[idx]:
                out_dice[cls].append(dice_list[idx][cls])

    # 4. Final Aggregation: Calculate mean Dice per class, ignoring NaNs
    out_dice_mean = [
        np.array(out_dice[cls]).mean() if len(out_dice[cls]) > 0 else np.nan
        for cls in range(cfg.model.out_channels - 1)
    ]

    # Return overall means across all valid classes
    return (
        global_val_loss,
        np.nanmean(out_dice_mean),
    )


def train_epoch_ddp(
    model,
    ema_model,
    train_loader,
    val_loader,
    optimizer,
    scaler,
    loss_function,
    post_label,
    post_pred,
    device,
    cfg,
    args,
    global_step,
    global_rank,
    dice_val_best,
    out_dir,
):
    model.train() # Set to training mode (enables dropout, tracks batch norm stats)

    for step_idx, batch in enumerate(train_loader):
        x, y = (batch["image"].to(device), batch["label"].to(device))

        # Update the learning rate every step based on our schedule
        current_lr = exp_lr_scheduler_with_warmup(
            optimizer,
            global_step,
            warmup_steps=cfg.training.warmup_steps,
            max_steps=cfg.training.max_iterations,
        )

        # Automatic Mixed Precision (AMP) block
        # Casts operations to float16 where safe to do so, drastically reducing memory usage and speeding up math.
        with torch.amp.autocast("cuda"):
            logit_map = model(x)
            loss = loss_function(logit_map, y)
            # Gradient Accumulation: Divide loss by accumulation steps.
            # This simulates a larger batch size by accumulating gradients over multiple 
            # forward passes before performing an optimization step.
            loss = loss / cfg.training.accumulation_steps

        # Scales the loss to prevent underflow (gradients becoming 0) in float16 precision
        scaler.scale(loss).backward()

        # Step the optimizer only after we have accumulated enough gradients
        if ((step_idx + 1) % cfg.training.accumulation_steps == 0) or (
            step_idx + 1 == len(train_loader)
        ):
            # Gradient Clipping: Unscale gradients first, then cap their maximum norm.
            # This prevents "exploding gradients", a common issue in 3D medical image segmentation.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update weights and scaler, then reset gradients for the next batch
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update EMA variables if we are using them
            if ema_model is not None:
                update_ema_variables(
                    model, ema_model, cfg.training.ema_alpha, global_step
                )

        # Logging (Only rank 0 logs to avoid duplicate prints/MLflow entries)
        if global_rank == 0 and (global_step == 1 or global_step % 50 == 0):
            display_loss = loss.item() * cfg.training.accumulation_steps
            print(
                f"Step: {global_step}/{cfg.training.max_iterations} | Loss: {display_loss:.4f} | LR: {current_lr:.6f}",
                flush=True,
            )
            mlflow.log_metric(
                f"fold_{args.fold}_train_loss", display_loss, step=global_step
            )

        # Validation Logic - Triggered at specific intervals or at the very end
        if (
            global_step % cfg.training.eval_num == 0 and global_step != 0
        ) or global_step == cfg.training.max_iterations:
            
            # If we trained an EMA model, we evaluate using its smoothed weights instead of the active model
            eval_target_model = ema_model if ema_model is not None else model

            if global_rank == 0:
                print(
                    f"\n--- Running Validation for Fold {args.fold} at Step {global_step} ---",
                    flush=True,
                )

            global_val_loss, global_dice = validation_ddp(
                eval_target_model,
                val_loader,
                loss_function,
                post_label,
                post_pred,
                device,
                cfg,
                args
            )

            # Save metrics and checkpoints
            if global_rank == 0:
                mlflow.log_metric(f"fold_{args.fold}_lr", current_lr, step=global_step)
                mlflow.log_metric(
                    f"fold_{args.fold}_val_loss", global_val_loss, step=global_step
                )
                mlflow.log_metric(
                    f"fold_{args.fold}_val_dice", global_dice, step=global_step
                )

                # Save the complete state so training can be resumed later if interrupted
                checkpoint_data = {
                    "step": global_step,
                    "model_state_dict": model.module.state_dict(),
                    "ema_model_state_dict": ema_model.module.state_dict() if ema_model else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_dice": dice_val_best,
                }
                torch.save(
                    checkpoint_data, os.path.join(out_dir, "latest_checkpoint.pth")
                )

                # Keep track of the best performing model
                if global_dice > dice_val_best:
                    dice_val_best = global_dice
                    checkpoint_data["best_dice"] = dice_val_best
                    torch.save(
                        checkpoint_data, os.path.join(out_dir, "best_metric_model.pth")
                    )
                    print(
                        f"Model Saved! New Best Dice: {dice_val_best:.4f}\n", flush=True
                    )
                else:
                    print(
                        f"Validation complete. Current Dice: {global_dice:.4f} (Best: {dice_val_best:.4f})\n",
                        flush=True,
                    )

            # Switch back to training mode after validation is complete
            model.train()

        # Ensure all GPUs wait for each other here before moving to the next step
        dist.barrier()
        global_step += 1

        # Break early if max iterations hit
        if global_step >= cfg.training.max_iterations:
            break

    return global_step, dice_val_best


# ==========================================
# Main
# ==========================================

def main(args, cfg):
    # Initialize the Distributed backend. NCCL is the standard/fastest backend for NVIDIA GPUs.
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))

    # -------------------------------------------------------------
    # LOCAL RANK vs GLOBAL RANK EXPLANATION:
    # -------------------------------------------------------------
    # In Distributed Data Parallel (DDP), a separate Python process is spawned for EACH GPU (or Node).
    # 
    # - "global_rank" (RANK) is the absolute, unique ID of the process across ALL machines.
    # - "local_rank" (LOCAL_RANK) is the ID of the process relative to the CURRENT machine.
    #
    # Example for 2 compute nodes (machines), each with 4 GPUs (8 GPUs total):
    # Node 0: GPUs [0, 1, 2, 3] -> local_ranks: [0, 1, 2, 3] | global_ranks: [0, 1, 2, 3]
    # Node 1: GPUs [0, 1, 2, 3] -> local_ranks: [0, 1, 2, 3] | global_ranks: [4, 5, 6, 7]
    # -------------------------------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    
    # We log system metrics (like CPU/GPU usage) using the global rank so we don't 
    # overwrite logs from other machines that share the same local_ranks.
    mlflow.system_metrics.set_system_metrics_node_id(f"rank_{global_rank}")

    # Bind the current process to a specific GPU on this specific machine.
    # If we didn't use local_rank here, all processes on a machine might try to 
    # cram their data onto GPU 0, causing out-of-memory errors.
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Ensures reproducibility across runs (seeds RNGs for Python, Numpy, and PyTorch)
    set_determinism(cfg.misc.seed)

    # Output directory setup (handled only by global rank 0 to prevent file write collisions across the entire cluster)
    out_dir = Path(args.output_model) / f"fold_{args.fold}"
    if global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving models for Fold {args.fold} to: {out_dir}", flush=True)

    train_files, val_files = get_files_from_csv(
        args.input_data, args.split_csv, args.fold
    )
    train_transforms, val_transforms = get_transforms(**cfg.transforms)

    # We use local_rank in the cache directory name so each GPU on the machine gets its own cache folder. 
    # This prevents multiple processes on the same machine from trying to read/write the exact same cache files simultaneously.
    cache_dir = Path(f"/tmp/monai_cache_rank_{local_rank}_fold_{args.fold}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # PersistentDataset dramatically speeds up training by executing the deterministic 
    # transforms once and saving the output to disk. Subsequent epochs load the pre-processed data.
    train_ds = PersistentDataset(
        data=train_files, transform=train_transforms, cache_dir=cache_dir
    )
    val_ds = PersistentDataset(
        data=val_files, transform=val_transforms, cache_dir=cache_dir
    )

    # 1. Samplers
    # DistributedSampler splits the dataset so each GPU gets a unique subset of the data per epoch.
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.train_batch_size, # Often 1 for 3D volumes due to memory constraints
        shuffle=False, # Shuffle is handled by the sampler
        num_workers=cfg.training.train_num_workers,
        pin_memory=True, # Speeds up host-to-device (CPU to GPU) transfers
        sampler=train_sampler,
        collate_fn=train_collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.val_batch_size,
        shuffle=False,
        num_workers=cfg.training.val_num_workers,
        pin_memory=True,
        sampler=val_sampler
    )

    # 2. Main Model
    model = get_segresnet(**cfg.model).to(device)
    
    # Wrap the model in DDP. 
    # device_ids and output_device must be set to the local_rank so PyTorch knows 
    # exactly which physical GPU on the machine this specific DDP instance is managing.
    model = DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    # 3. EMA Model Initialization
    ema_model = None
    if cfg.training.get("use_ema", True):
        ema_model = get_segresnet(**cfg.model).to(device)
        # The EMA model also needs to be wrapped in DDP and tied to the local_rank.
        ema_model = DistributedDataParallel(
            ema_model, device_ids=[local_rank], output_device=local_rank
        )
        # Detach EMA model from the computational graph (it's updated manually, not via backprop)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        if global_rank == 0:
            print("EMA Model initialized for evaluation.", flush=True)

    # DiceFocalLoss combines the overlap-based optimization of Dice with 
    # the class-imbalance handling of Focal Loss.
    loss_function = DiceFocalLoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        gamma=2.0,
        lambda_dice=0.5,
        lambda_focal=0.5,
        alpha=0.75,
    )
    
    # Enables cuDNN auto-tuner to find the fastest convolution algorithms for your specific hardware
    torch.backends.cudnn.benchmark = True
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    # Post-processing transforms for metrics calculation
    post_label = AsDiscrete(to_onehot=2)
    post_pred = AsDiscrete(argmax=True, to_onehot=2)

    # Initialize the gradient scaler for Automatic Mixed Precision (AMP)
    scaler = torch.amp.GradScaler("cuda")

    global_step = 0
    dice_val_best = 0.0

    # 4. Resume Logic updated to handle EMA
    if args.checkpoint and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Load weights into the main module (handling the DDP wrapper properly)
        if "model_state_dict" in checkpoint:
            model.module.load_state_dict(checkpoint["model_state_dict"])
            if (
                ema_model
                and "ema_model_state_dict" in checkpoint
                and checkpoint["ema_model_state_dict"]
            ):
                ema_model.module.load_state_dict(checkpoint["ema_model_state_dict"])
            
            # If strictly resuming training (not just fine-tuning or testing), load optimizer states
            if args.resume:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                global_step = checkpoint["step"]
                dice_val_best = checkpoint.get("best_dice", 0.0)
        else:
            model.module.load_state_dict(checkpoint)

    # Ensure all GPUs start at the exact same global step if resuming
    step_tensor = torch.tensor([global_step], dtype=torch.long, device=device)
    dist.broadcast(step_tensor, src=0)
    global_step = step_tensor.item()

    optimizer.zero_grad()
    epochs_completed = global_step // len(train_loader)

    if global_rank == 0:
        print("Starting training loop...", flush=True)

    # Main Training Loop
    while global_step < cfg.training.max_iterations:
        # Crucial for DDP: The sampler needs to know the epoch to shuffle data properly across GPUs
        train_sampler.set_epoch(epochs_completed)

        global_step, dice_val_best = train_epoch_ddp(
            model=model,
            ema_model=ema_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_function=loss_function,
            post_label=post_label,
            post_pred=post_pred,
            device=device,
            cfg=cfg,
            args=args,
            global_step=global_step,
            global_rank=global_rank,
            dice_val_best=dice_val_best,
            out_dir=out_dir,
        )
        epochs_completed += 1

    # Clean up the DDP process group when training finishes
    dist.destroy_process_group()


if __name__ == "__main__":
    # Standard argparse setup combined with OmegaConf for flexible hierarchical config management
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument("--split_csv", type=str, default="cv_splits.csv")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=str, default="config/train_config.yaml")

    args, unknown_args = parser.parse_known_args()
    
    # Load base YAML config, then merge any unknown CLI arguments as overrides
    cfg = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_dotlist(unknown_args)
    cfg = OmegaConf.merge(cfg, cli_conf)

    main(args, cfg)