import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from ptflops import get_model_complexity_info
from torchvision import transforms
from torchmetrics.segmentation import MeanIoU
from time import perf_counter
from tqdm import tqdm


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


def overlay(image, mask, color=(0, 0, 255), alpha=0.5):
    mask_np = np.array(mask).squeeze().astype(bool)
    overlay = np.array(image).copy().astype(np.uint)
    overlay[mask_np] = (1 - alpha) * overlay[mask_np] + alpha * np.array(color)
    return overlay.astype(np.uint8)


def show_visualization(model, dataset, device='cpu', start_idx=0, skip_idx=1, num_samples=20, threshold=0.5):
    model.to(device)
    model.eval()

    for i in range(num_samples):
        image_tensor, mask_tensor = dataset[start_idx + (i * skip_idx)]
        input_tensor = image_tensor.unsqueeze(0).to(device)  # add batch dimension

        with torch.no_grad():
            output = model(input_tensor)

        pred_mask = (torch.sigmoid(output) > threshold).float().squeeze(0).cpu()

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].set_title("Original Image + Mask")
        axs[0].imshow(overlay(get_img_from_tensor(image_tensor), mask_tensor))
        axs[0].axis("off")
        axs[1].set_title("Predicted Mask")
        axs[1].imshow(overlay(get_img_from_tensor(image_tensor), pred_mask))
        axs[1].axis("off")
        plt.show()
    


def measure_global_iou(model, dataloader, device='cuda'):
    print(f"Measuring global IoU on '{device}'...")

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = 'cpu'
    
    model.to(device)
    model.eval()

    miou = MeanIoU(num_classes=2, include_background=True, per_class=True, input_format='index').to(device)

    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images, masks = images.to(device), masks.to(device)

            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5)

            miou.update(preds.long(), masks.long())

    global_miou = miou.compute()

    bg_iou = global_miou[0].item()
    fg_iou = global_miou[1].item()

    print(f"Background IoU: {bg_iou:.4f}")
    print(f"Foreground IoU: {fg_iou:.4f}")

    return bg_iou, fg_iou


def measure_macs_params(model, input_size=(3, 96, 96)):
    macs, params = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
    
    print(f"MACs:   {macs / 1e6:.1f} MMACs")
    print(f"Params: {params / 1e3:.1f} k")

    return macs, params


def measure_inference_time(model, input_size=(3, 96, 96), device='cuda', iterations=500):
    print(f"Measuring inference time on '{device}' over {iterations} iterations...")

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = 'cpu'
    
    model.to(device)
    model.eval()

    input_tensor = torch.randn(*input_size).to(device)

    if device == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

    # Warm-up
    with torch.no_grad():
        for _ in range(50):
            _ = model(input_tensor)
        
    # Measurement
    t = []
    with torch.no_grad():
        for _ in range(iterations):
            if device == 'cuda':
                start.record()
                _ = model(input_tensor)
                end.record()
                torch.cuda.synchronize()
                t_delta = start.elapsed_time(end)  # milliseconds
            else:
                start = perf_counter()
                _ = model(input_tensor)
                end = perf_counter()
                t_delta = (end - start) * 1000  # milliseconds

            t.append(t_delta)
                
    t_avg = np.mean(t)
    t_std = np.std(t)
    t_p99 = np.percentile(t, 99)

    print(f"Avg Inference Time (per image): {t_avg:.2f} ms ± {t_std:.2f} ms ")
    print(f"P99 Inference Time (per image): {t_p99:.2f} ms ")

    return t_avg, t_std, t_p99


def benchmark(model, dataloader=None, device='cpu', input_size=(1, 3, 96, 96), iterations=500):
    print("=== Benchmark Summary ===")
    macs, params = measure_macs_params(model, input_size[1:])
    t_avg, t_std, t_p99 = measure_inference_time(model, input_size, device, iterations)
    bg_iou, fg_iou = measure_global_iou(model, dataloader, device)
    print("=========================\n")

    return {
        "macs": macs,
        "params": params,
        "inference_time_avg_ms": t_avg,
        "inference_time_std_ms": t_std,
        "inference_time_p99_ms": t_p99,
        "bg_iou": bg_iou,
        "fg_iou": fg_iou
    }