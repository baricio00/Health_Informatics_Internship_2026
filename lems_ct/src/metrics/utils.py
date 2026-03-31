import torch
from . import metrics
import numpy as np


def calculate_distance(label_pred, label_true, spacing, C, percentage=95):

    label_pred = label_pred.cpu().numpy()
    label_true = label_true.cpu().numpy()
    
    if torch.is_tensor(spacing):
        spacing = spacing.cpu().numpy()
    spacing = spacing.flatten()

    ASD_list = np.zeros(C-1)
    HD_list = np.zeros(C-1)

    for i in range(C-1):
        tmp_surface = metrics.compute_surface_distances(label_true[i+1].astype(bool), label_pred[i+1].astype(bool), spacing)
        dis_gt_to_pred, dis_pred_to_gt = metrics.compute_average_surface_distance(tmp_surface)
        ASD_list[i] = (dis_gt_to_pred + dis_pred_to_gt) / 2 

        HD = metrics.compute_robust_hausdorff(tmp_surface, percentage)
        HD_list[i] = HD

    return ASD_list, HD_list


def calculate_dice_split(pred, target, C, block_size=64*64*64):
    
    pred = pred.view(C, -1).t()
    target = target.view(C, -1).t()
    
    assert pred.shape[0] == target.shape[0]
    N = pred.shape[0]
    total_sum = torch.zeros(C).to(pred.device)
    total_intersection = torch.zeros(C).to(pred.device)
    
    split_num = N // block_size
    for i in range(split_num):
        dice, intersection, summ = calculate_dice(pred[i*block_size:(i+1)*block_size, :], target[i*block_size:(i+1)*block_size, :], C)
        total_intersection += intersection
        total_sum += summ
    if N % block_size != 0:
        dice, intersection, summ = calculate_dice(pred[(i+1)*block_size:, :], target[(i+1)*block_size:, :], C)
        total_intersection += intersection
        total_sum += summ

    dice = 2 * total_intersection / (total_sum + 1e-5)

    return dice, total_intersection, total_sum


def calculate_dice(pred, target, C): 

    target = target.long()
    pred = pred.long()
    
    target_mask = target
    pred_mask = pred

    intersection = pred_mask * target_mask
    summ = pred_mask + target_mask

    intersection = intersection.sum(0).type(torch.float32)
    summ = summ.sum(0).type(torch.float32)
    
    summ += 1e-5 
    dice = 2 * intersection / summ

    return dice, intersection, summ