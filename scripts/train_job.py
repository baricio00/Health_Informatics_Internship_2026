import os
import sys
import math
import time
import argparse
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
from lems_ct.src.metrics.utils import calculate_dice_split, calculate_distance
from scripts.lcc_postprocessing import lcc_one_hot_after_argmax


# ==========================================
# SOTA Utility Functions
# ==========================================


def update_ema_variables(model, ema_model, alpha, global_step):
    """Smoothly updates EMA model weights."""
    alpha = min((1 - 1 / (global_step + 1)), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    for ema_buffer, m_buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(m_buffer)


def exp_lr_scheduler_with_warmup(optimizer, step, warmup_steps, max_steps):
    """Exponential LR scheduler with warmup phase."""
    for g in optimizer.param_groups:
        g.setdefault("base_lr", g["lr"])

    if warmup_steps and 0 <= step <= warmup_steps:
        lr_mult = math.exp(10.0 * (float(step) / float(warmup_steps) - 1.0))
        if step == warmup_steps:
            lr_mult = 1.0
    else:
        lr_mult = (1.0 - step / max_steps) ** 0.9

    for g in optimizer.param_groups:
        g["lr"] = g["base_lr"] * lr_mult
    return optimizer.param_groups[0]["lr"]


@torch.no_grad()
def concat_all_gather(tensor):
    """Gathers tensors from all GPUs and concatenates them."""
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


# ==========================================
# Core Logic
# ==========================================


def _calculate_case_metrics(label_pred, labels, spacing, cfg):
    """Calculate per-class metrics for a single discretized prediction."""
    tmp_ASD_list, tmp_HD_list = calculate_distance(
        label_pred, labels, spacing, cfg.model.out_channels
    )
    tmp_ASD_list = np.clip(np.nan_to_num(tmp_ASD_list, nan=500), 0, 500)
    tmp_HD_list = np.clip(np.nan_to_num(tmp_HD_list, nan=500), 0, 500)
    tmp_dice_list, _, _ = calculate_dice_split(
        label_pred.view(-1, 1), labels.view(-1, 1), cfg.model.out_channels
    )
    return tmp_dice_list.unsqueeze(0), np.expand_dims(tmp_ASD_list, axis=0), np.expand_dims(tmp_HD_list, axis=0)


def _aggregate_validation_metrics(dice_list, ASD_list, HD_list, unique_labels_list, out_channels):
    """Average foreground metrics only for classes present in the ground truth."""
    out_dice = [[] for _ in range(out_channels - 1)]
    out_ASD = [[] for _ in range(out_channels - 1)]
    out_HD = [[] for _ in range(out_channels - 1)]

    for idx in range(len(dice_list)):
        for cls in range(out_channels - 1):
            if cls + 1 in unique_labels_list[idx]:
                out_dice[cls].append(dice_list[idx][cls])
                out_ASD[cls].append(ASD_list[idx][cls])
                out_HD[cls].append(HD_list[idx][cls])

    out_dice_mean = [
        np.array(out_dice[cls]).mean() if out_dice[cls] else np.nan
        for cls in range(out_channels - 1)
    ]
    out_ASD_mean = [
        np.array(out_ASD[cls]).mean() if out_ASD[cls] else np.nan
        for cls in range(out_channels - 1)
    ]
    out_HD_mean = [
        np.array(out_HD[cls]).mean() if out_HD[cls] else np.nan
        for cls in range(out_channels - 1)
    ]
    return np.nanmean(out_dice_mean), np.nanmean(out_HD_mean), np.nanmean(out_ASD_mean)


def validation_ddp(
    model, val_loader, loss_function, post_label, post_pred_raw, post_pred_lcc, device, cfg, args
):
    """
    Validate the deployed pipeline while retaining raw metrics for auditability.

    The loss is always computed on untouched logits. LCC is applied only after
    argmax/to-one-hot discretization and is used for checkpoint selection.
    """
    model.eval()

    raw_dice_list, raw_ASD_list, raw_HD_list = [], [], []
    lcc_dice_list, lcc_ASD_list, lcc_HD_list = [], [], []
    unique_labels_list = []

    local_val_loss = 0.0
    local_steps = 0
    global_rank = int(os.environ.get("RANK", 0))

    if global_rank == 0:
        print("\nEvaluating raw and LCC-postprocessed predictions...", flush=True)

    with torch.no_grad():
        for step, val_batch in enumerate(val_loader, start=1):
            if global_rank == 0:
                print(f"\nValidating volume {step}/{len(val_loader)}...", flush=True)

            torch.cuda.synchronize()
            t0 = time.time()
            val_inputs = val_batch["image"].to(device).float()
            val_labels = val_batch["label"].to(device).long()
            spacing = np.array(cfg.transforms.target_spacing)

            with torch.amp.autocast("cuda"):
                val_outputs = sliding_window_inference(
                    val_inputs,
                    cfg.transforms.roi_size,
                    cfg.inference.sw_batch_size,
                    model.module,
                    overlap=cfg.inference.overlap,
                    mode=cfg.inference.mode,
                )
                local_val_loss += loss_function(val_outputs, val_labels).item()
                local_steps += 1

            torch.cuda.synchronize()
            t1 = time.time()
            if global_rank == 0:
                print(f"  -> Inference: {t1-t0:.2f}s", flush=True)

            val_labels_convert = [post_label(x) for x in decollate_batch(val_labels)]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_raw = [post_pred_raw(x) for x in val_outputs_list]
            val_output_lcc = [
                lcc_one_hot_after_argmax(
                    x.unsqueeze(0),
                    cfg.model.out_channels,
                )[0]
                for x in val_outputs_list
            ]

            labels = val_labels_convert[0].cpu().bool()
            label_pred_raw = val_output_raw[0].cpu()
            label_pred_lcc = val_output_lcc[0].cpu()

            unique_labels = torch.unique(labels).cpu().numpy()
            unique_labels = np.pad(unique_labels, (100 - len(unique_labels), 0), "constant", constant_values=0)

            raw_dice, raw_asd, raw_hd = _calculate_case_metrics(label_pred_raw, labels, spacing, cfg)
            lcc_dice, lcc_asd, lcc_hd = _calculate_case_metrics(label_pred_lcc, labels, spacing, cfg)
            unique_labels = np.expand_dims(unique_labels, axis=0)

            raw_dice = concat_all_gather(raw_dice.cuda(device)).cpu().numpy()[:, 1:]
            lcc_dice = concat_all_gather(lcc_dice.cuda(device)).cpu().numpy()[:, 1:]
            unique_labels = concat_all_gather(torch.from_numpy(unique_labels).cuda(device)).cpu().numpy()
            raw_asd = concat_all_gather(torch.from_numpy(raw_asd).cuda(device)).cpu().numpy()
            raw_hd = concat_all_gather(torch.from_numpy(raw_hd).cuda(device)).cpu().numpy()
            lcc_asd = concat_all_gather(torch.from_numpy(lcc_asd).cuda(device)).cpu().numpy()
            lcc_hd = concat_all_gather(torch.from_numpy(lcc_hd).cuda(device)).cpu().numpy()

            for idx in range(len(lcc_dice)):
                raw_dice_list.append(raw_dice[idx])
                raw_ASD_list.append(raw_asd[idx])
                raw_HD_list.append(raw_hd[idx])
                lcc_dice_list.append(lcc_dice[idx])
                lcc_ASD_list.append(lcc_asd[idx])
                lcc_HD_list.append(lcc_hd[idx])
                unique_labels_list.append(unique_labels[idx])

            if global_rank == 0:
                print(f"  -> Raw + LCC metrics: {time.time()-t1:.2f}s", flush=True)

    local_loss_avg = torch.tensor([local_val_loss / max(1, local_steps)], device=device)
    dist.all_reduce(local_loss_avg, op=dist.ReduceOp.SUM)
    global_val_loss = (local_loss_avg / dist.get_world_size()).item()

    world_size = dist.get_world_size()
    dataset_len = len(val_loader.dataset)
    padding_size = 0 if (dataset_len % world_size) == 0 else world_size - (dataset_len % world_size)
    for _ in range(padding_size):
        if lcc_dice_list:
            raw_dice_list.pop(); raw_ASD_list.pop(); raw_HD_list.pop()
            lcc_dice_list.pop(); lcc_ASD_list.pop(); lcc_HD_list.pop()
            unique_labels_list.pop()

    raw_metrics = _aggregate_validation_metrics(
        raw_dice_list, raw_ASD_list, raw_HD_list, unique_labels_list, cfg.model.out_channels
    )
    lcc_metrics = _aggregate_validation_metrics(
        lcc_dice_list, lcc_ASD_list, lcc_HD_list, unique_labels_list, cfg.model.out_channels
    )

    # Return LCC metrics first: checkpoint selection must match deployed inference.
    return global_val_loss, *lcc_metrics, *raw_metrics

def train_epoch_ddp(
    model,
    ema_model,
    train_loader,
    val_loader,
    optimizer,
    scaler,
    loss_function,
    post_label,
    post_pred_raw,
    post_pred_lcc,
    device,
    cfg,
    args,
    global_step,
    global_rank,
    dice_val_best,
    out_dir,
):
    model.train()

    for step_idx, batch in enumerate(train_loader):
        x, y = (batch["image"].to(device), batch["label"].to(device))

        # LR Warmup and Scheduler per step
        current_lr = exp_lr_scheduler_with_warmup(
            optimizer,
            global_step,
            warmup_steps=cfg.training.warmup_steps,
            max_steps=cfg.training.max_iterations,
        )

        with torch.amp.autocast("cuda"):
            logit_map = model(x)
            loss = loss_function(logit_map, y)
            loss = loss / cfg.training.accumulation_steps

        scaler.scale(loss).backward()

        if ((step_idx + 1) % cfg.training.accumulation_steps == 0) or (
            step_idx + 1 == len(train_loader)
        ):
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update EMA variables
            if ema_model is not None:
                update_ema_variables(
                    model, ema_model, cfg.training.ema_alpha, global_step
                )

        if global_rank == 0 and (global_step == 1 or global_step % 50 == 0):
            display_loss = loss.item() * cfg.training.accumulation_steps
            print(
                f"Step: {global_step}/{cfg.training.max_iterations} | Loss: {display_loss:.4f} | LR: {current_lr:.6f}",
                flush=True,
            )
            mlflow.log_metric(
                f"fold_{args.fold}_train_loss", display_loss, step=global_step
            )

        # Validation
        if (
            global_step % cfg.training.eval_num == 0 and global_step != 0
        ) or global_step == cfg.training.max_iterations:
            # Use EMA model for evaluation if available
            eval_target_model = ema_model if ema_model is not None else model

            if global_rank == 0:
                print(
                    f"\n--- Running Validation for Fold {args.fold} at Step {global_step} ---",
                    flush=True,
                )

            (
                global_val_loss,
                global_dice,
                global_hd95,
                global_asd,
                global_dice_raw,
                global_hd95_raw,
                global_asd_raw,
            ) = validation_ddp(
                eval_target_model,
                val_loader,
                loss_function,
                post_label,
                post_pred_raw,
                post_pred_lcc,
                device,
                cfg,
                args
            )

            if global_rank == 0:
                mlflow.log_metric(f"fold_{args.fold}_lr", current_lr, step=global_step)
                mlflow.log_metric(
                    f"fold_{args.fold}_val_loss", global_val_loss, step=global_step
                )
                mlflow.log_metric(
                    f"fold_{args.fold}_val_dice", global_dice, step=global_step
                )
                mlflow.log_metric(
                    f"fold_{args.fold}_val_hd95", global_hd95, step=global_step
                )
                mlflow.log_metric(
                    f"fold_{args.fold}_val_ASD", global_asd, step=global_step
                )
                mlflow.log_metric(
                    f"fold_{args.fold}_val_dice_raw", global_dice_raw, step=global_step
                )
                mlflow.log_metric(
                    f"fold_{args.fold}_val_hd95_raw", global_hd95_raw, step=global_step
                )
                mlflow.log_metric(
                    f"fold_{args.fold}_val_ASD_raw", global_asd_raw, step=global_step
                )

                # Unpack and save both main and EMA dicts
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

            model.train()

        dist.barrier()
        global_step += 1

        if global_step >= cfg.training.max_iterations:
            break

    return global_step, dice_val_best


# ==========================================
# Main Initialization
# ==========================================


def main(args, cfg):
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    mlflow.system_metrics.set_system_metrics_node_id(f"rank_{global_rank}")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    set_determinism(cfg.misc.seed)

    out_dir = Path(args.output_model) / f"fold_{args.fold}"
    if global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving models for Fold {args.fold} to: {out_dir}", flush=True)

    train_files, val_files = get_files_from_csv(
        args.input_data, args.split_csv, args.fold
    )
    train_transforms, val_transforms = get_transforms(**cfg.transforms)

    cache_dir = Path(f"./monai_cache_rank_{local_rank}_fold_{args.fold}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PersistentDataset(
        data=train_files, transform=train_transforms, cache_dir=cache_dir
    )
    val_ds = PersistentDataset(
        data=val_files, transform=val_transforms, cache_dir=cache_dir
    )

    # 1. Samplers
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=val_sampler,
    )

    # 2. Main Model
    model = get_segresnet(**cfg.model).to(device)
    model = DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    # 3. EMA Model Initialization
    ema_model = None
    if cfg.training.get("use_ema", True):
        ema_model = get_segresnet(**cfg.model).to(device)
        ema_model = DistributedDataParallel(
            ema_model, device_ids=[local_rank], output_device=local_rank
        )
        for p in ema_model.parameters():
            p.requires_grad_(False)
        if global_rank == 0:
            print("EMA Model initialized for evaluation.", flush=True)

    loss_function = DiceFocalLoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        gamma=2.0,
        lambda_dice=0.5,
        lambda_focal=0.5,
        alpha=0.75,
    )
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    post_label = AsDiscrete(to_onehot=cfg.model.out_channels)
    post_pred_raw = AsDiscrete(argmax=True, to_onehot=cfg.model.out_channels)
    post_pred_lcc = None

    scaler = torch.amp.GradScaler("cuda")

    global_step = 0
    dice_val_best = 0.0

    # 4. Resume Logic updated to handle EMA
    if args.checkpoint and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)

        if "model_state_dict" in checkpoint:
            model.module.load_state_dict(checkpoint["model_state_dict"])
            if (
                ema_model
                and "ema_model_state_dict" in checkpoint
                and checkpoint["ema_model_state_dict"]
            ):
                ema_model.module.load_state_dict(checkpoint["ema_model_state_dict"])
            if args.resume:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                global_step = checkpoint["step"]
                dice_val_best = checkpoint.get("best_dice", 0.0)
        else:
            model.module.load_state_dict(checkpoint)

    step_tensor = torch.tensor([global_step], dtype=torch.long, device=device)
    dist.broadcast(step_tensor, src=0)
    global_step = step_tensor.item()

    optimizer.zero_grad()
    epochs_completed = global_step // len(train_loader)

    if global_rank == 0:
        print("Starting training loop...", flush=True)

    while global_step < cfg.training.max_iterations:
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
            post_pred_raw=post_pred_raw,
            post_pred_lcc=post_pred_lcc,
            device=device,
            cfg=cfg,
            args=args,
            global_step=global_step,
            global_rank=global_rank,
            dice_val_best=dice_val_best,
            out_dir=out_dir,
        )
        epochs_completed += 1

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument("--split_csv", type=str, default="data/cv_splits_qc.csv")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=str, default="config/train_config.yaml")

    args, unknown_args = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_dotlist(unknown_args)
    cfg = OmegaConf.merge(cfg, cli_conf)

    main(args, cfg)
