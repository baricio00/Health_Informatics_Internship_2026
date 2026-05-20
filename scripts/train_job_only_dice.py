import os
import sys
import time
import csv
import hashlib
import json
import argparse
import platform
from contextlib import nullcontext
import mlflow
from omegaconf import OmegaConf
from pathlib import Path
from datetime import timedelta
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from monai.utils import set_determinism
from monai.losses import (
    DiceLoss,
    DiceCELoss,
    DiceFocalLoss,
    HausdorffDTLoss,
    TverskyLoss,
)

# from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.data import PersistentDataset, DataLoader, decollate_batch
from monai.transforms import AsDiscrete

from lems_ct.src.models.model import get_segresnet
from lems_ct.src.utils.transforms import get_transforms
from lems_ct.src.utils.data import get_files_from_csv, resolve_data_root, resolve_local_path
from lems_ct.src.utils.misc import (
    update_ema_variables,
    exp_lr_scheduler_with_warmup,
    concat_all_gather,
    train_collate_fn
)
from lems_ct.src.metrics.utils import calculate_dice_split


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def get_model_state(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def load_model_state(model, state_dict):
    target_model = model.module if hasattr(model, "module") else model
    target_model.load_state_dict(state_dict)


def inference_model(model):
    return model.module if hasattr(model, "module") else model


def autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def release_cuda_memory(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gather_tensor(tensor):
    if get_world_size() == 1:
        return tensor
    return concat_all_gather(tensor)


def reduce_mean_loss(local_loss_avg):
    if get_world_size() == 1:
        return local_loss_avg.item()
    dist.all_reduce(local_loss_avg, op=dist.ReduceOp.SUM)
    return (local_loss_avg / get_world_size()).item()


def barrier():
    if is_distributed():
        dist.barrier()


def is_rosetta_or_intel_python():
    return sys.platform == "darwin" and platform.machine() == "x86_64"


def select_device(requested_device, distributed):
    if requested_device != "auto":
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("ERROR: CUDA was requested, but torch.cuda is not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            details = (
                f"python_arch={platform.machine()}, "
                f"mps_built={torch.backends.mps.is_built()}, "
                f"mps_available={torch.backends.mps.is_available()}"
            )
            if is_rosetta_or_intel_python():
                details += ". Use a native arm64 Python environment, not an x86_64/Rosetta environment."
            raise SystemExit(
                "ERROR: MPS was requested, but torch.backends.mps is not available "
                f"({details})."
            )
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return device

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")

    if distributed:
        return torch.device("cpu")

    if torch.backends.mps.is_available() and not is_rosetta_or_intel_python():
        return torch.device("mps")

    return torch.device("cpu")


def init_distributed_if_needed():
    required_env = {"RANK", "WORLD_SIZE", "LOCAL_RANK"}
    distributed = required_env.issubset(os.environ)
    if not distributed:
        return False

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, timeout=timedelta(hours=2))
    return True


def create_output_dir(out_dir):
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise SystemExit(
            f"ERROR: Cannot create output directory '{out_dir}'. "
            "Use a writable local path such as './outputs' when running outside Azure/notebook containers."
        ) from None


def normalize_local_paths(args, global_rank):
    original_input_data = Path(args.input_data)
    original_output_model = Path(args.output_model)

    args.input_data = str(resolve_local_path(args.input_data))
    args.output_model = str(resolve_local_path(args.output_model))

    if global_rank == 0:
        if Path(args.input_data) != original_input_data:
            print(f"Resolved local input_data: {original_input_data} -> {args.input_data}", flush=True)
        if Path(args.output_model) != original_output_model:
            print(f"Resolved local output_model: {original_output_model} -> {args.output_model}", flush=True)


def canonical_loss_name(loss_name):
    aliases = {
        "dice": "dice",
        "dice_ce": "dice_ce",
        "dicece": "dice_ce",
        "dice_focal": "dice_focal",
        "dicefocal": "dice_focal",
        "hausdorff": "hausdorff",
        "haussdorf": "hausdorff",
        "hausdorff_dt": "hausdorff",
        "tversky": "tversky",
    }
    normalized = str(loss_name).lower().replace("-", "_")
    if normalized not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise SystemExit(f"ERROR: Unknown loss '{loss_name}'. Valid losses: {valid}")
    return aliases[normalized]


class DownsampledHausdorffDTLoss(nn.Module):
    def __init__(
        self,
        downsample_factor=2,
        lambda_hausdorff=1.0,
        lambda_dice=0.0,
        **kwargs,
    ):
        super().__init__()
        self.downsample_factor = max(1, int(downsample_factor))
        self.lambda_hausdorff = float(lambda_hausdorff)
        self.lambda_dice = float(lambda_dice)
        self.hausdorff = HausdorffDTLoss(**kwargs)
        self.dice = DiceLoss(**kwargs) if self.lambda_dice > 0 else None

    def forward(self, input_tensor, target_tensor):
        if self.downsample_factor > 1:
            spatial_size = [
                max(1, dim // self.downsample_factor)
                for dim in input_tensor.shape[2:]
            ]
            hausdorff_input = F.interpolate(
                input_tensor,
                size=spatial_size,
                mode="trilinear",
                align_corners=False,
            )
            hausdorff_target = F.interpolate(
                target_tensor.float(),
                size=spatial_size,
                mode="nearest",
            ).long()
        else:
            hausdorff_input = input_tensor
            hausdorff_target = target_tensor

        hausdorff_input = hausdorff_input.float()
        hausdorff_device = hausdorff_input.device
        if hausdorff_device.type == "mps":
            # MONAI's distance transform creates float64 tensors, which MPS does not support.
            # Keep the model on MPS, but compute the distance-transform part on CPU.
            hausdorff_input = hausdorff_input.cpu()
            hausdorff_target = hausdorff_target.cpu()

        loss = self.lambda_hausdorff * self.hausdorff(
            hausdorff_input, hausdorff_target
        )
        if hausdorff_device.type == "mps":
            loss = loss.to(hausdorff_device)
        if self.dice is not None:
            loss = loss + self.lambda_dice * self.dice(input_tensor, target_tensor)
        return loss


def common_loss_kwargs():
    return {
        "include_background": False,
        "to_onehot_y": True,
        "softmax": True,
    }


def get_loss_function(cfg):
    loss_name = canonical_loss_name(cfg.training.get("loss", "dice_focal"))
    common = common_loss_kwargs()

    if loss_name == "dice":
        return loss_name, DiceLoss(**common)
    if loss_name == "dice_ce":
        return loss_name, DiceCELoss(**common)
    if loss_name == "dice_focal":
        return loss_name, DiceFocalLoss(
            **common,
            gamma=2.0,
            lambda_dice=0.5,
            lambda_focal=0.5,
            alpha=0.75,
        )
    if loss_name == "hausdorff":
        return loss_name, DownsampledHausdorffDTLoss(
            downsample_factor=cfg.training.get("hausdorff_downsample", 2),
            lambda_hausdorff=cfg.training.get("hausdorff_lambda", 1.0),
            lambda_dice=cfg.training.get("hausdorff_lambda_dice", 0.0),
            **common,
        )
    if loss_name == "tversky":
        return loss_name, TverskyLoss(**common, alpha=0.3, beta=0.7)

    raise AssertionError(f"Unhandled loss: {loss_name}")


def get_validation_loss_function(cfg, train_loss_name, train_loss_function):
    validation_loss_name = str(
        cfg.training.get(
            "hausdorff_validation_loss" if train_loss_name == "hausdorff" else "validation_loss",
            "dice" if train_loss_name == "hausdorff" else train_loss_name,
        )
    )
    validation_loss_name = canonical_loss_name(validation_loss_name)

    if validation_loss_name == train_loss_name:
        return validation_loss_name, train_loss_function

    validation_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    validation_cfg.training.loss = validation_loss_name
    return get_loss_function(validation_cfg)


def format_sequence(values):
    return "x".join(str(value) for value in values)


def append_csv_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file_obj:
        json.dump(json_safe(data), file_obj, indent=2, sort_keys=True)


def metric_context(cfg, args, loss_name):
    return {
        "fold": args.fold,
        "loss": loss_name,
        "base_lr": float(cfg.training.lr),
        "weight_decay": float(cfg.training.weight_decay),
        "roi_size": format_sequence(cfg.transforms.roi_size),
        "target_spacing": format_sequence(cfg.transforms.target_spacing),
        "max_iterations": int(cfg.training.max_iterations),
        "train_batch_size": int(cfg.training.get("train_batch_size", 0)),
        "accumulation_steps": int(cfg.training.get("accumulation_steps", 1)),
        "use_ema": bool(cfg.training.get("use_ema", True)),
        "hausdorff_downsample": int(cfg.training.get("hausdorff_downsample", 1)),
        "hausdorff_lambda": float(cfg.training.get("hausdorff_lambda", 1.0)),
        "hausdorff_lambda_dice": float(cfg.training.get("hausdorff_lambda_dice", 0.0)),
        "hausdorff_validation_loss": str(
            cfg.training.get("hausdorff_validation_loss", "")
        ),
    }


def write_run_config(out_dir, cfg, args, loss_name):
    OmegaConf.save(config=cfg, f=str(Path(out_dir) / "resolved_config.yaml"))
    write_json(
        Path(out_dir) / "hparams.json",
        {
            **metric_context(cfg, args, loss_name),
            "input_data": args.input_data,
            "split_csv": args.split_csv,
            "output_model": args.output_model,
            "device": args.device,
        },
    )


def log_run_params(cfg, args, loss_name):
    mlflow.log_params(metric_context(cfg, args, loss_name))


def cache_dir_for_run(out_dir, local_rank):
    run_key = hashlib.sha1(str(Path(out_dir).resolve()).encode()).hexdigest()[:12]
    return Path(f"/tmp/monai_cache_{run_key}_rank_{local_rank}")


def record_validation_metrics(out_dir, cfg, args, loss_name, metrics):
    row = {
        **metric_context(cfg, args, loss_name),
        **metrics,
    }
    row = json_safe(row)
    append_csv_row(Path(out_dir) / "metrics.csv", row)
    write_json(Path(out_dir) / "summary.json", row)


def is_cuda_oom(exc):
    message = str(exc).lower()
    oom_error = getattr(torch, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_error) or "cuda out of memory" in message


def cuda_memory_context():
    if not torch.cuda.is_available():
        return {}

    device_idx = torch.cuda.current_device()
    divisor = 1024**3
    return {
        "cuda_device": torch.cuda.get_device_name(device_idx),
        "cuda_allocated_gb": torch.cuda.memory_allocated(device_idx) / divisor,
        "cuda_reserved_gb": torch.cuda.memory_reserved(device_idx) / divisor,
        "cuda_max_allocated_gb": torch.cuda.max_memory_allocated(device_idx) / divisor,
        "cuda_max_reserved_gb": torch.cuda.max_memory_reserved(device_idx) / divisor,
    }


def record_training_failure(args, cfg, exc, failure_type):
    global_rank = int(os.environ.get("RANK", 0))
    if global_rank != 0:
        return

    memory_context = cuda_memory_context()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    loss_name = canonical_loss_name(cfg.training.get("loss", "dice_focal"))
    out_dir = Path(args.output_model) / f"fold_{args.fold}"
    create_output_dir(out_dir)
    write_run_config(out_dir, cfg, args, loss_name)

    row = {
        **metric_context(cfg, args, loss_name),
        "status": failure_type,
        "failure_type": failure_type,
        "failed": True,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "best_val_dice": "",
        "val_dice": "",
        "val_loss": "",
        "train_loss": "",
        "step": "",
        **memory_context,
    }
    row = json_safe(row)
    append_csv_row(Path(out_dir) / "metrics.csv", row)
    write_json(Path(out_dir) / "summary.json", row)

    try:
        mlflow.set_tag("training_status", failure_type)
        mlflow.set_tag("failure_type", failure_type)
        mlflow.log_metric(f"fold_{args.fold}_failed", 1)
    except Exception as mlflow_exc:
        print(f"WARNING: Could not log failure to MLflow: {mlflow_exc}", flush=True)


# ==========================================
# Core Logic
# ==========================================

def validation_ddp(
    model, val_loader, validation_loss_function, post_label, post_pred, device, cfg, args
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
    max_validation_batches = int(cfg.training.get("max_validation_batches", 0) or 0)
    if max_validation_batches < 0:
        raise ValueError("training.max_validation_batches must be >= 0")
    validation_batches = (
        min(len(val_loader), max_validation_batches)
        if max_validation_batches
        else len(val_loader)
    )

    with torch.no_grad():
        for step, val_batch in enumerate(val_loader, start=1):
            if max_validation_batches and step > max_validation_batches:
                break
            if global_rank == 0:
                print(f"\nValidating volume {step}/{validation_batches}...", flush=True)

            # Synchronize CUDA before measuring time to ensure accurate benchmarking
            synchronize_device(device)
            t0 = time.time()
                
            val_inputs = val_batch["image"].to(device).float()
            val_labels = val_batch["label"].to(device).long()

            # Use Automatic Mixed Precision (AMP) for faster inference
            with autocast_context(device):
                # sliding_window_inference is crucial for medical imaging. 3D volumes 
                # are often too large to fit in GPU memory at once. This function breaks 
                # the volume into overlapping patches, infers on each, and stitches them back together.
                val_outputs = sliding_window_inference(
                    val_inputs,
                    cfg.transforms.roi_size,
                    cfg.inference.sw_batch_size,
                    inference_model(model),
                    overlap=cfg.inference.overlap,
                    mode=cfg.inference.mode,
                )
                local_val_loss += validation_loss_function(val_outputs, val_labels).item()
                local_steps += 1

            synchronize_device(device)
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
            tmp_dice_list = gather_tensor(tmp_dice_list.to(device))
            unique_labels = (
                gather_tensor(torch.from_numpy(unique_labels).to(device))
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

            del (
                val_inputs,
                val_labels,
                val_outputs,
                val_labels_list,
                val_labels_convert,
                val_outputs_list,
                val_output_convert,
                label_pred,
                labels,
                tmp_dice_list,
                unique_labels,
            )
            release_cuda_memory(device)

    # 1. Average local validation loss across all GPUs
    local_loss_avg = torch.tensor([local_val_loss / max(1, local_steps)], device=device)
    global_val_loss = reduce_mean_loss(local_loss_avg)

    # 2. Remove padded duplicate samples from DistributedSampler
    # DistributedSampler pads the dataset to ensure every GPU gets the exact same number of batches.
    # We must remove these duplicate "dummy" samples before calculating our final metrics.
    world_size = get_world_size()
    dataset_len = len(val_loader.dataset)
    padding_size = 0
    if not max_validation_batches:
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
    validation_loss_function,
    post_label,
    post_pred,
    device,
    cfg,
    args,
    global_step,
    global_rank,
    dice_val_best,
    out_dir,
    loss_name,
    validation_loss_name,
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
        with autocast_context(device):
            logit_map = model(x)
            loss = loss_function(logit_map, y)
            # Gradient Accumulation: Divide loss by accumulation steps.
            # This simulates a larger batch size by accumulating gradients over multiple 
            # forward passes before performing an optimization step.
            loss = loss / cfg.training.accumulation_steps

        display_loss = loss.item() * cfg.training.accumulation_steps

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
            optimizer.zero_grad(set_to_none=True)

            # Update EMA variables if we are using them
            if ema_model is not None:
                update_ema_variables(
                    model, ema_model, cfg.training.ema_alpha, global_step
                )

        # Logging (Only rank 0 logs to avoid duplicate prints/MLflow entries)
        step_number = global_step + 1
        if global_rank == 0 and (step_number == 1 or step_number % 50 == 0):
            print(
                f"Step: {step_number}/{cfg.training.max_iterations} | Loss: {display_loss:.4f} | LR: {current_lr:.6f}",
                flush=True,
            )
            mlflow.log_metric(
                f"fold_{args.fold}_train_loss", display_loss, step=step_number
            )

        # Validation Logic - Triggered at specific intervals or at the very end
        if (
            step_number % cfg.training.eval_num == 0 and step_number != 0
        ) or step_number == cfg.training.max_iterations:
            
            # If we trained an EMA model, we evaluate using its smoothed weights instead of the active model
            eval_target_model = ema_model if ema_model is not None else model

            if global_rank == 0:
                print(
                    f"\n--- Running Validation for Fold {args.fold} at Step {step_number} ---",
                    flush=True,
                )

            global_val_loss, global_dice = validation_ddp(
                eval_target_model,
                val_loader,
                validation_loss_function,
                post_label,
                post_pred,
                device,
                cfg,
                args
            )
            release_cuda_memory(device)

            # Save metrics and checkpoints
            if global_rank == 0:
                mlflow.log_metric(f"fold_{args.fold}_lr", current_lr, step=step_number)
                mlflow.log_metric(
                    f"fold_{args.fold}_val_loss", global_val_loss, step=step_number
                )
                mlflow.log_metric(
                    f"fold_{args.fold}_val_dice", global_dice, step=step_number
                )

                # Save the complete state so training can be resumed later if interrupted
                is_best = global_dice > dice_val_best
                best_dice_after_step = global_dice if is_best else dice_val_best
                checkpoint_data = {
                    "step": step_number,
                    "model_state_dict": get_model_state(model),
                    "ema_model_state_dict": get_model_state(ema_model) if ema_model else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_dice": best_dice_after_step,
                    "loss_name": loss_name,
                    "validation_loss_name": validation_loss_name,
                }
                torch.save(
                    checkpoint_data, os.path.join(out_dir, "latest_checkpoint.pth")
                )

                # Keep track of the best performing model
                if is_best:
                    dice_val_best = best_dice_after_step
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

                mlflow.log_metric(
                    f"fold_{args.fold}_best_val_dice", dice_val_best, step=step_number
                )

                record_validation_metrics(
                    out_dir,
                    cfg,
                    args,
                    loss_name,
                    {
                        "step": step_number,
                        "train_loss": display_loss,
                        "val_loss": global_val_loss,
                        "val_loss_name": validation_loss_name,
                        "val_dice": global_dice,
                        "best_val_dice": dice_val_best,
                        "lr": current_lr,
                        "is_best": int(is_best),
                        "latest_checkpoint": str(Path(out_dir) / "latest_checkpoint.pth"),
                        "best_checkpoint": str(Path(out_dir) / "best_metric_model.pth"),
                    },
                )

            # Switch back to training mode after validation is complete
            model.train()

        # Ensure all GPUs wait for each other here before moving to the next step
        barrier()
        global_step += 1

        # Break early if max iterations hit
        if global_step >= cfg.training.max_iterations:
            break

        del x, y, logit_map, loss

    return global_step, dice_val_best


# ==========================================
# Main
# ==========================================

def main(args, cfg):
    distributed = init_distributed_if_needed()

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
    normalize_local_paths(args, global_rank)
    
    # We log system metrics (like CPU/GPU usage) using the global rank so we don't 
    # overwrite logs from other machines that share the same local_ranks.
    mlflow.system_metrics.set_system_metrics_node_id(f"rank_{global_rank}")

    # Bind the current process to a specific GPU on this specific machine.
    # If we didn't use local_rank here, all processes on a machine might try to 
    # cram their data onto GPU 0, causing out-of-memory errors.
    device = select_device(args.device, distributed)
    if global_rank == 0:
        mode = "distributed" if distributed else "single-process"
        print(f"Running in {mode} mode on device: {device}", flush=True)
    
    # Ensures reproducibility across runs (seeds RNGs for Python, Numpy, and PyTorch)
    set_determinism(cfg.misc.seed)

    # Output directory setup (handled only by global rank 0 to prevent file write collisions across the entire cluster)
    out_dir = Path(args.output_model) / f"fold_{args.fold}"
    if global_rank == 0:
        create_output_dir(out_dir)
        print(f"Saving models for Fold {args.fold} to: {out_dir}", flush=True)

    train_files, val_files = get_files_from_csv(
        args.input_data, args.split_csv, args.fold
    )
    if not train_files or not val_files:
        expected_root = resolve_data_root(args.input_data)
        raise SystemExit(
            "ERROR: No training/validation scans found. "
            f"Checked: {expected_root}. "
            "Expected patient folders containing CT_LATE.nii.gz and registration_mask.nii.gz."
        )

    train_transforms, val_transforms = get_transforms(**cfg.transforms)

    # We use local_rank in the cache directory name so each GPU on the machine gets its own cache folder. 
    # This prevents multiple processes on the same machine from trying to read/write the exact same cache files simultaneously.
    cache_dir = cache_dir_for_run(out_dir, local_rank)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if global_rank == 0:
        print(f"Using MONAI cache dir: {cache_dir}", flush=True)

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
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.train_batch_size, # Often 1 for 3D volumes due to memory constraints
        shuffle=not distributed,
        num_workers=cfg.training.train_num_workers,
        pin_memory=device.type == "cuda", # Speeds up host-to-device (CPU to GPU) transfers
        sampler=train_sampler,
        collate_fn=train_collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.val_batch_size,
        shuffle=False,
        num_workers=cfg.training.val_num_workers,
        pin_memory=device.type == "cuda",
        sampler=val_sampler
    )

    # 2. Main Model
    model = get_segresnet(**cfg.model).to(device)
    
    # Wrap the model in DDP. 
    # device_ids and output_device must be set to the local_rank so PyTorch knows 
    # exactly which physical GPU on the machine this specific DDP instance is managing.
    if distributed:
        if device.type == "cuda":
            model = DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank
            )
        else:
            model = DistributedDataParallel(model)

    # 3. EMA Model Initialization
    ema_model = None
    if cfg.training.get("use_ema", True):
        ema_model = get_segresnet(**cfg.model).to(device)
        # The EMA model also needs to be wrapped in DDP and tied to the local_rank.
        if distributed:
            if device.type == "cuda":
                ema_model = DistributedDataParallel(
                    ema_model, device_ids=[local_rank], output_device=local_rank
                )
            else:
                ema_model = DistributedDataParallel(ema_model)
        # Detach EMA model from the computational graph (it's updated manually, not via backprop)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        if global_rank == 0:
            print("EMA Model initialized for evaluation.", flush=True)

    loss_name, loss_function = get_loss_function(cfg)
    validation_loss_name, validation_loss_function = get_validation_loss_function(
        cfg, loss_name, loss_function
    )
    if global_rank == 0:
        write_run_config(out_dir, cfg, args, loss_name)
        log_run_params(cfg, args, loss_name)
        print(f"Using loss: {loss_name}", flush=True)
        if validation_loss_name != loss_name:
            print(f"Using validation loss: {validation_loss_name}", flush=True)
    
    # Enables cuDNN auto-tuner to find the fastest convolution algorithms for your specific hardware
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    # Post-processing transforms for metrics calculation
    post_label = AsDiscrete(to_onehot=2)
    post_pred = AsDiscrete(argmax=True, to_onehot=2)

    # Initialize the gradient scaler for Automatic Mixed Precision (AMP)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    global_step = 0
    dice_val_best = 0.0

    # 4. Resume Logic updated to handle EMA
    if args.checkpoint and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Load weights into the main module (handling the DDP wrapper properly)
        if "model_state_dict" in checkpoint:
            load_model_state(model, checkpoint["model_state_dict"])
            if (
                ema_model
                and "ema_model_state_dict" in checkpoint
                and checkpoint["ema_model_state_dict"]
            ):
                load_model_state(ema_model, checkpoint["ema_model_state_dict"])
            
            # If strictly resuming training (not just fine-tuning or testing), load optimizer states
            if args.resume:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                global_step = checkpoint["step"]
                dice_val_best = checkpoint.get("best_dice", 0.0)
        else:
            load_model_state(model, checkpoint)

    # Ensure all GPUs start at the exact same global step if resuming
    step_tensor = torch.tensor([global_step], dtype=torch.long, device=device)
    if distributed:
        dist.broadcast(step_tensor, src=0)
    global_step = step_tensor.item()

    optimizer.zero_grad(set_to_none=True)
    epochs_completed = global_step // len(train_loader)

    if global_rank == 0:
        print("Starting training loop...", flush=True)

    # Main Training Loop
    while global_step < cfg.training.max_iterations:
        # Crucial for DDP: The sampler needs to know the epoch to shuffle data properly across GPUs
        if train_sampler is not None:
            train_sampler.set_epoch(epochs_completed)

        global_step, dice_val_best = train_epoch_ddp(
            model=model,
            ema_model=ema_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_function=loss_function,
            validation_loss_function=validation_loss_function,
            post_label=post_label,
            post_pred=post_pred,
            device=device,
            cfg=cfg,
            args=args,
            global_step=global_step,
            global_rank=global_rank,
            dice_val_best=dice_val_best,
            out_dir=out_dir,
            loss_name=loss_name,
            validation_loss_name=validation_loss_name,
        )
        epochs_completed += 1

    # Clean up the DDP process group when training finishes
    if distributed:
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
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device for local runs. Azure/torchrun DDP still uses LOCAL_RANK when CUDA is available.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help=(
            "Record recoverable trial failures such as CUDA OOM in summary.json "
            "and exit 0 so sweep pipelines can continue."
        ),
    )

    args, unknown_args = parser.parse_known_args()
    
    # Load base YAML config, then merge any unknown CLI arguments as overrides
    cfg = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_dotlist(unknown_args)
    cfg = OmegaConf.merge(cfg, cli_conf)

    try:
        main(args, cfg)
    except Exception as exc:
        if args.continue_on_error and is_cuda_oom(exc):
            print(
                "CUDA OOM recorded as a failed sweep trial. "
                "Exiting 0 because --continue_on_error was set.",
                flush=True,
            )
            record_training_failure(args, cfg, exc, "cuda_oom")
            sys.exit(0)
        raise
