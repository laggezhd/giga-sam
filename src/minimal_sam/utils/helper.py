import numpy as np
import torch
import tqdm

from PIL import Image
from ptflops import get_model_complexity_info
from torchvision import transforms
from time import perf_counter


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


def compute_batch_counts(logits: torch.Tensor, targets: torch.Tensor, threshold=0.5):
    assert torch.all((targets == 0) | (targets == 1)), "Targets must be binary (0 or 1)"
    assert logits.shape == targets.shape, "Logits and targets must have the same shape"

    preds = (torch.sigmoid(logits) > threshold)
    targets = (targets > 0.5) 

    # Summing over all dimensions (Batch, C, H, W) gives the total pixel count
    intersection = (preds & targets).long().sum().item()
    union = (preds | targets).long().sum().item()
    
    return intersection, union


def benchmark_model(model_name, dataset_name, model, dataloader, device):
    
    total_intersection = 0.0
    total_union = 0.0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)
    timings = []

    model.to(device)
    model.eval()

    print(f"Warming up {device}...")
    dummy = torch.randn(1, 3, 96, 96).to(device)
    for _ in range(20):
        _ = model(dummy)

    with torch.no_grad():
        print(f"\nEvaluating model: {model_name} on {dataset_name} dataset...")
        for images, masks in tqdm(dataloader):
            images, masks = images.to(device), masks.to(device)
    
            start_event.record()

            # Forward pass
            logits = model(images)
            
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
            timings.append(elapsed_time)

            # Get intersection and union for this batch (works for batch_size >= 1)
            # We do NOT calculate IoU here, just raw pixel counts
            inter, union = compute_batch_counts(logits, masks)
            
            total_intersection += inter
            total_union += union

    # Compute metric only once at the very end
    fps = 1000.0 / np.mean(timings) * dataloader.batch_size
    global_iou = total_intersection / (total_union + 1e-6)
    
    print(f"--- Results for {model_name} ---")
    print(f"Processed {len(dataloader.dataset)} images.\n")
    print(f"Mean Inference Time: {np.mean(timings):.2f} ms ± {np.std(timings):.2f} ms per batch")
    print(f"P99 Inference Time:  {np.percentile(timings, 99):.2f} ms per batch")
    print(f"FPS (Throughput):    {fps:.2f} images/sec\n")
    print(f"{model_name} Global mIoU on {dataset_name} val set: {global_iou:.4f}")

def measure_model_macs_params(model, input_size=(1, 3, 96, 96)):
    flops, params = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
    
    print(f"MACs:   {flops / 1e6:.1f} MMACs")
    print(f"Params: {params / 1e3:.1f} k")

    return flops, params

def measure_inference_time(model, input_size=(1, 3, 96, 96), device='cuda', iterations=100):
    print(f"Measuring inference time on device: {device}")

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
    

