###################################################################################################
#
# Dataset definition for ai8x-training
# 96x96 images cropped from COCO dataset
# Cyril Scherrer, 2025
#
###################################################################################################

"""
MinimalSam dataset
"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


import ai8x

import os
import json
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from pycocotools.coco import COCO


"""
Custom image dataset class
"""
class MinimalSamDataset(Dataset):
    def __init__(self, annotation_file: str, img_dir: str, img_size: int, transform=None, filtered_annotation_file=False):
        super().__init__()

        self.img_dir = img_dir
        self.img_size = img_size
        
        self.coco = COCO(annotation_file)
        self.anns = []

        self.transform = transform

        if filtered_annotation_file:
            # load from json file
            with open(filtered_annotation_file, "r") as f:
                self.filtered_anns = json.load(f)
            return
        
        # === Pre-filtering annotations without valid mask ===
        self.count_no_mask = 0 #8
        self.count_center_not_in_mask = 0 #106336
        self.count_invalid_crop = 0 #115905
        self.count_high_mask_area = 0 #175600
        
        self.filtered_anns = []
        self.anns = [ann for ann in self.coco.loadAnns(self.coco.getAnnIds()) if ann.get("iscrowd", 0) == 0]


        for ann in tqdm(self.anns):
            mask = self.coco.annToMask(ann)

            ys, xs = np.where(mask > 0)
            
            # does mask exist?
            if len(xs) == 0:
                self.count_no_mask += 1
                continue

            min_x, max_x = xs.min(), xs.max()
            min_y, max_y = ys.min(), ys.max()
            mask_center_x = (min_x + max_x) // 2
            mask_center_y = (min_y + max_y) // 2

            # is center in mask?
            if not mask[mask_center_y, mask_center_x]:
                self.count_center_not_in_mask += 1
                continue

            center_x, center_y = mask_center_x, mask_center_y
            
            # define crop box
            left = center_x - self.img_size // 2
            top  = center_y - self.img_size // 2
            right  = left + self.img_size
            bottom = top + self.img_size

            # check if crop box is valid
            if left < 0 or top < 0 or right > mask.shape[1] or bottom > mask.shape[0]:
                self.count_invalid_crop += 1
                continue

            # check if the cropped mask area occupies more than 60% of the crop area
            crop_area = self.img_size * self.img_size
            mask_area = np.sum(mask[top:bottom, left:right] > 0)
            if mask_area / crop_area > 0.6:
                self.count_high_mask_area += 1
                continue

            self.filtered_anns.append(ann)

    def __len__(self):
        return len(self.filtered_anns)
    
    def __getitem__(self, index):
        ann = self.filtered_anns[index]
        img_id = ann['image_id']
        img = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img['file_name'])

        image = Image.open(img_path).convert("RGB")
        mask = self.coco.annToMask(ann)

        cropped_image, cropped_mask = self._crop(image, mask)

        image_tensor = self.transform(cropped_image)
        mask_tensor = torch.tensor(np.array(cropped_mask), dtype=torch.long)

        return image_tensor, mask_tensor
    
    def _crop(self, image, mask):
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
        cropped_img = image.crop((left, top, right, bottom)).resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        cropped_mask = Image.fromarray(mask[top:bottom, left:right]).resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)

        return cropped_img, cropped_mask  #, center_x, center_y, left, top

"""
Dataloader function
"""
def minimalsam_get_datasets(data, load_train=False, load_test=False):
   
    (data_dir, args) = data
    # data_dir = data

    transform = transforms.Compose([
        transforms.ToTensor(), # maps RGB to [0,1]
        ai8x.normalize(args=args), # maps [0,1] to [-1,1]
    ])

    if load_train:
        annotation_file = os.path.join(data_dir, "annotations/instances_train2017.json")
        img_dir = os.path.join(data_dir, "train2017")
        img_size = 96
        filtered_annotation_file = os.path.join(data_dir, "annotations/filtered_anns_96x96_train2017_coco.json")
        train_dataset = MinimalSamDataset(annotation_file, img_dir, img_size, transform, filtered_annotation_file=filtered_annotation_file)

    else:
        train_dataset = None

    if load_test:
        annotation_file = os.path.join(data_dir, "annotations/instances_val2017.json")
        img_dir = os.path.join(data_dir, "val2017")
        img_size = 96
        filtered_annotation_file = os.path.join(data_dir, "annotations/filtered_anns_96x96_val2017_coco.json")
        test_dataset = MinimalSamDataset(annotation_file, img_dir, img_size, transform, filtered_annotation_file=filtered_annotation_file)

    else:
        test_dataset = None

    return train_dataset, test_dataset


"""
Dataset description
"""
datasets = [
    {
        'name': 'minimalsam_dataset',
        'input': (3, 96, 96),
        'output': (0, 1), # binary segmentation
        'loader': minimalsam_get_datasets,
    }
]