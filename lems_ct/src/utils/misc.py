import math
import torch
import torch.distributed as dist

def update_ema_variables(model, ema_model, alpha, global_step):
    """
    Smoothly updates Exponential Moving Average (EMA) model weights.
    
    Why use EMA? Instead of using the exact weights of the model from the final training step, 
    EMA maintains a moving average of the weights throughout training. This often leads to 
    more robust models that generalize better to unseen data, acting as a form of ensembling over time.
    """
    # The alpha parameter controls the decay rate. We use a step-based warmup 
    # to allow faster updates at the very beginning of training.
    alpha = min((1 - 1 / (global_step + 1)), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # ema_weight = (alpha * ema_weight) + ((1 - alpha) * current_weight)
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    for ema_buffer, m_buffer in zip(ema_model.buffers(), model.buffers()):
        # Buffers (like batch norm running statistics) are copied directly
        ema_buffer.copy_(m_buffer)


def exp_lr_scheduler_with_warmup(optimizer, step, warmup_steps, max_steps):
    """
    Exponential Learning Rate scheduler with a linear warmup phase.
    
    Warmup prevents early training instability (exploding gradients) by gradually increasing 
    the learning rate from near-zero to the base learning rate before decaying it.
    """
    for g in optimizer.param_groups:
        g.setdefault("base_lr", g["lr"])

    if warmup_steps and 0 <= step <= warmup_steps:
        # Exponential curve scaling up to 1.0 at the end of warmup
        lr_mult = math.exp(10.0 * (float(step) / float(warmup_steps) - 1.0))
        if step == warmup_steps:
            lr_mult = 1.0
    else:
        # Polynomial decay after warmup finishes
        lr_mult = (1.0 - step / max_steps) ** 0.9

    for g in optimizer.param_groups:
        g["lr"] = g["base_lr"] * lr_mult
    return optimizer.param_groups[0]["lr"]


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Gathers tensors from all GPUs in a Distributed Data Parallel (DDP) setup and concatenates them.
    
    In multi-GPU training, each GPU calculates metrics on its own chunk of data. 
    To get the true global metric (e.g., overall validation Dice score), we must 
    gather the predictions from every GPU onto one place.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)