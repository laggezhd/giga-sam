import json
import numpy as np
import os
import torch

from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class MinimalSamDataset(Dataset):
    """
    A minimal dataset class for loading images and their corresponding masks from COCO/LVIS annotations.
    
    Each annotation results in a valid cropped image-mask pair based on specific the following criteria:
        1. The target image and mask size is `img_size` x `img_size`.
        2. The annotation must have a mask (non-empty).
        3. The center point of the mask must be within the mask itself 
        4. The crop box around the mask center must be valid (within image bounds).
        5. The mask to crop area ratio must not exceed `mask_ratio`.

    Thus, a single COCO image can contribute multiple samples to the dataset if it contains multiple valid annotations!
    This makes the dataset significantly larger in images compared to the standard COCO images (e.g. COCO val2017 has 
    5,000 images. After filtering, the new dataset contains ~17,000 samples).
    
    Note that once an image is transformed to a tensor, we normalize it using the ImageNet mean and std.

    Args:
        img_size (int): Target image size for cropping.
        dataset_dir (str): Path to dataset directory.
        annotation_file (str): Path to COCO/LVIS annotation file.
        filtered_anns_file (str): Path to the pre-filtered annotations file. If None is provided, filtering is performed.
        mask_ratio (float): Maximum allowed ratio of mask area to crop area when filtering.
    """

    def __init__(self,
        img_size: int,
        dataset_dir: str,
        annotation_file: str, 
        filtered_anns_file: str = None,
        mask_ratio=0.6
        ): 
        super().__init__()

        assert Path(annotation_file).exists(), f"Annotation file {annotation_file} does not exist."
        assert Path(dataset_dir).exists(), f"Dataset directory {dataset_dir} does not exist."

        self.dataset_dir = dataset_dir
        self.img_size = img_size

        self.coco = COCO(annotation_file)
        self.anns = [ann for ann in self.coco.loadAnns(self.coco.getAnnIds()) if ann.get("iscrowd", 0) == 0]

        if filtered_anns_file and os.path.exists(filtered_anns_file):
            print(f"Loading pre-filtered annotations from {filtered_anns_file}")
            with open(filtered_anns_file, "r") as f:
                self.filtered_anns = json.load(f)
        else:
            print("Filtering annotations...")
            self.filtered_anns = self.filter_anns(mask_ratio)
            print(f"Saving filtered annotations to {filtered_anns_file}")
            with open(filtered_anns_file, "w") as f:
                json.dump(self.filtered_anns, f)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
        ])

    def __len__(self):
        return len(self.filtered_anns)

    def __getitem__(self, index):
        ann = self.filtered_anns[index]
        img = self.coco.loadImgs(ann['image_id'])[0]

        img_file = Path('/'.join(img['coco_url'].split('/')[-2:]))
        img_path = Path(self.dataset_dir).joinpath(img_file)

        image = Image.open(img_path).convert("RGB")
        mask  = self.coco.annToMask(ann)

        cropped_image, cropped_mask = self.crop(image, mask)

        image_tensor = self.transform(cropped_image)
        mask_tensor  = torch.tensor(np.array(cropped_mask), dtype=torch.float32).unsqueeze(0) 

        return image_tensor, mask_tensor

    def filter_anns(self, mask_ratio: float) -> list:
        filtered_anns = []

        count_no_mask = 0               #8, 8(64x64)
        count_center_not_in_mask = 0    #106336, 106336(64x64)
        count_invalid_crop = 0          #115905, 64327(64x64)
        count_high_mask_area = 0        #175600, 256834(64x64)

        for ann in tqdm(self.anns):
            mask = self.coco.annToMask(ann)

            ys, xs = np.where(mask > 0)

            # does mask exist?
            if len(xs) == 0:
                count_no_mask += 1
                continue

            min_x, max_x = xs.min(), xs.max()
            min_y, max_y = ys.min(), ys.max()
            mask_center_x = (min_x + max_x) // 2
            mask_center_y = (min_y + max_y) // 2

            # is center in mask?
            if not mask[mask_center_y, mask_center_x]:
                count_center_not_in_mask += 1
                continue

            center_x, center_y = mask_center_x, mask_center_y

            # define crop box
            left = center_x - self.img_size // 2
            top  = center_y - self.img_size // 2
            right  = left + self.img_size
            bottom = top + self.img_size

            # check if crop box is valid
            if left < 0 or top < 0 or right > mask.shape[1] or bottom > mask.shape[0]:
                count_invalid_crop += 1
                continue

            # check if the cropped mask area occupies more than mask_ratio of the crop area
            crop_area = self.img_size * self.img_size
            mask_area = np.sum(mask[top:bottom, left:right] > 0)
            if mask_area / crop_area > mask_ratio:
                count_high_mask_area += 1
                continue

            filtered_anns.append(ann)

        print(f"Shrunk dataset by {len(filtered_anns)/len(self.anns)*100:.2f}%")
        print(f"Count no mask:            {count_no_mask}")
        print(f"Count center not in mask: {count_center_not_in_mask}")
        print(f"Count invalid crop:       {count_invalid_crop}")
        print(f"Count high mask area:     {count_high_mask_area}")
        print(f"Dataset original samples: {len(self.anns)}")
        print(f"Dataset filtered samples: {len(filtered_anns)}")

        return filtered_anns

    def crop(self, image, mask):
        ys, xs = np.where(mask > 0)

        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        mask_center_x = (min_x + max_x) // 2
        mask_center_y = (min_y + max_y) // 2

        center_x, center_y = mask_center_x, mask_center_y

        # define crop box
        left = center_x - self.img_size // 2
        top  = center_y - self.img_size // 2
        right  = left + self.img_size
        bottom = top + self.img_size

        # could leave out resizing here
        cropped_img  = image.crop((left, top, right, bottom)).resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        cropped_mask = Image.fromarray(mask[top:bottom, left:right]).resize((self.img_size, self.img_size), Image.Resampling.NEAREST)

        return cropped_img, cropped_mask