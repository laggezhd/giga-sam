import torch
import torch.nn.functional as F


def bce_dice_loss(pred_mask, gt_mask):
    bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
    
    # Dice Loss needs probabilities
    pred_probs = torch.sigmoid(pred_mask)
    
    intersection = (pred_probs * gt_mask).sum(dim=(1, 2, 3))
    union = pred_probs.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3))
    
    dice = 1 - ((2 * intersection + 1e-6) / (union + 1e-6)).mean()
    
    return bce + dice