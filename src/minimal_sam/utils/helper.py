import numpy as np
import torch

from PIL import Image
from torchvision import transforms


def get_img_from_tensor(tensor: torch.Tensor) -> Image.Image:
    """
    Args:
        tensor (torch.Tensor): Normalized (ImageNet) image tensor of shape (3, H, W).
    Returns:
        image (PIL.Image): Unnormalized PIL image of shape (H, W, 3) with pixel values in [0, 255] (RGB).
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(tensor.device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(tensor.device)
    unnormalized_tensor = (tensor * std + mean).clamp(0,1)
    transform = transforms.ToPILImage()
    return transform(unnormalized_tensor)

def overlay_mask(image, mask, color=(0, 0, 255), alpha=0.5):
    mask_np = np.array(mask).squeeze().astype(bool)
    overlay = np.array(image).copy().astype(np.uint)
    overlay[mask_np] = (1 - alpha) * overlay[mask_np] + alpha * np.array(color)
    return overlay.astype(np.uint8)