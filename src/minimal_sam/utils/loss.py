import torch
import torch.nn.functional as F


# def bce_dice_loss(pred_mask, gt_mask):
# pred_mask = torch.sigmoid(pred_mask)
# # idea: bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
# bce = F.binary_cross_entropy(pred_mask, gt_mask)
# intersection = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
# union = pred_mask.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3))
# dice = 1 - ((2 * intersection + 1e-6) / (union + 1e-6)).mean()
# return bce + dice


# IMPROVEMENT: use logits for BCE calculation, only use probs for Dice calculation
# This avoids numerical instability, see pyTorch docs for F.binary_cross_entropy_with_logits
def bce_dice_loss(pred_mask, gt_mask):
    # 1. Calculate BCE using the logits (raw model output) directly.
    # Do NOT apply torch.sigmoid() before this step.
    bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
    
    # 2. For Dice, we still need the probabilities, so we apply sigmoid here separately.
    pred_probs = torch.sigmoid(pred_mask)
    
    intersection = (pred_probs * gt_mask).sum(dim=(1, 2, 3))
    union = pred_probs.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3))
    
    dice = 1 - ((2 * intersection + 1e-6) / (union + 1e-6)).mean()
    
    return bce + dice