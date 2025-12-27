import torch
import numpy as np


def compute_iou(pred_mask, target_mask):
    pred_binary = (torch.sigmoid(pred_mask) > 0.5).cpu().numpy()
    target_binary = (torch.sigmoid(target_mask) > 0.5).cpu().numpy()

    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()
    return intersection / union if union > 0 else 1.0