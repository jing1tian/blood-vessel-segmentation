import os
from typing import Any
import cv2
import time
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.augmentations.geometric import functional as F
from albumentations.core.transforms_interface import to_tuple, DualTransform

from utils import binary2dist
from transforms import rotate_slice


group_shapes = {
    "kidney_1_dense": (2279, 1303, 912),
    "kidney_1_voi": (1397, 1928, 1928),
    "kidney_2": (2217, 1041, 1511),
    "kidney_3": (1035, 1706, 1510),
    "kidney_3_dense": (501, 1706, 1510),
    "kidney_3_sparse": (1035, 1706, 1510),
}


def get_train_transforms(image_size=512):
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Affine(scale={"x":(0.7, 1.3), "y":(0.7, 1.3)}, translate_percent={"x":(0, 0.1), "y":(0, 0.1)}, rotate=(-30, 30), shear=(-20, 20), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.2),
            A.MedianBlur(blur_limit=3, p=0.2),
        ], p=1.0),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, border_mode=1, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.1, border_mode=1, p=0.5)
        ], p=0.4),
        A.OneOf([
            A.Resize(image_size, image_size, cv2.INTER_LINEAR, p=1),
            A.Compose([
                RandomResize(image_size, image_size, scale_limit_x=0.5, scale_limit_y=0.5, p=1),
                A.PadIfNeeded(image_size, image_size, position="random", border_mode=cv2.BORDER_REPLICATE, p=1.0),
                A.RandomCrop(image_size, image_size, p=1.0)
            ], p=1.0),
        ], p=1.0),
        A.GaussNoise(var_limit=0.05, p=0.2),
    ])
    return transforms
    
def get_valid_transforms(image_size=512):
    transforms = A.Compose([
        A.Resize(image_size, image_size, cv2.INTER_LINEAR),
    ])
    return transforms
    
class HOADataset(Dataset):
    def __init__(
        self, 
        memmap_dir,
        groups,
        channels=1, 
        transforms=None, 
        mixup=0.0, 
        output_dist_maps=False, 
        normalize_dist_map=False,
        rotate_slice_prob=0.0,
        rotate_slice_angle_limit=30,
    ):
        
        # assert channels == 1 
        super().__init__()
        self.channels = channels
        self.transforms = transforms
        self.mixup = mixup
        self.output_dist_maps = output_dist_maps
        self.normalize_dist_map = normalize_dist_map
        self.memmap_dir = memmap_dir
        self.rotate_slice_prob = rotate_slice_prob
        self.rotate_slice_angle_limit = rotate_slice_angle_limit
        
        self.groups = groups.strip().split("|")
        self.metas = []
        for group in self.groups:
            shape = group_shapes[group.replace("_xz", "").replace("_zy", "")]
            num_slice = shape[0]
            if "_xz" in group: num_slice = shape[1]
            if "_zy" in group: num_slice = shape[2]
            for i in range(num_slice):
                self.metas.append({
                    "group": group,
                    "slice_id": i
                })
                        
    def __len__(self):
        return len(self.metas)
    
    def _load_image_and_mask(self, group, slice_id):
        # load memmap
        striped_group = group.replace('_xz', '').replace('_zy', '')
        volume = np.memmap(os.path.join(self.memmap_dir, f"{striped_group}.mmap"), dtype=np.uint16, shape=group_shapes[striped_group], mode="r")
        volume_mask = np.memmap(os.path.join(self.memmap_dir, f"{striped_group}_mask.mmap"), dtype=np.uint8, shape=group_shapes[striped_group], mode="r")
        if "_xz" in group:
            volume = volume.transpose((1, 2, 0))
            volume_mask = volume_mask.transpose((1, 2, 0))
        elif "_zy" in group:
            volume = volume.transpose((2, 0, 1))
            volume_mask = volume_mask.transpose((2, 0, 1))
            
        # random slice rotation
        if random.random() < self.rotate_slice_prob:
            try:
                angle = np.random.uniform(-self.rotate_slice_angle_limit, self.rotate_slice_angle_limit, size=3)
                image, mask = rotate_slice(volume, slice_id, angle, channels=self.channels, volume_mask=volume_mask)
                image = image.transpose((2, 1, 0)) # to [channels, height, width]
                mask = mask.transpose((2, 1, 0))
            except:
                # This is due to the numerical error of geometry3d
                # rarely happen.
                logger.warning("Rotate error. Skip Rotation!")
                if self.channels == 1:
                    image = volume[slice_id]
                    mask = volume_mask[slice_id]
                else:
                    image = volume[np.clip(range(slice_id-self.channels//2, slice_id+self.channels//2+1), 0, volume.shape[0]-1)]
                    mask = volume_mask[np.clip(range(slice_id-self.channels//2, slice_id+self.channels//2+1), 0, volume.shape[0]-1)]
        else:
            if self.channels == 1:
                image = volume[slice_id]
                mask = volume_mask[slice_id]
            else:
                image = volume[np.clip(range(slice_id-self.channels//2, slice_id+self.channels//2+1), 0, volume.shape[0]-1)]
                mask = volume_mask[np.clip(range(slice_id-self.channels//2, slice_id+self.channels//2+1), 0, volume.shape[0]-1)]
                
        image = image.astype(np.float32) / 65535.0
        mask = (mask>0).astype(np.float32)
        return image, mask

    def get_one_sample(self, idx):
        meta = self.metas[idx]
        group, slice_id = meta["group"], meta["slice_id"]
        image, mask = self._load_image_and_mask(group, slice_id)
        if len(image.shape) <= 2:
            image = np.tile(image[..., None], [1, 1, 3]) # gray to rgb
            mask = mask[None, ...]
        else:
            # random flip for channel dim
            if random.random() < 0.5:
                image = image[::-1]
                mask = mask[::-1]
            image = image.transpose(1, 2, 0)

        if self.transforms:
            data = self.transforms(image=image, masks=mask)
            image  = data['image']
            mask  = (np.stack([mask>0.5 for mask in data['masks']], axis=0)).astype(np.float32)

        if self.output_dist_maps:
            # haven't tried 3d binary2dist. maybe better?
            dist_map = torch.tensor(np.stack([binary2dist(m).astype(np.float32) for m in mask], axis=0))
        else:
            dist_map = None

        image = torch.tensor(np.transpose(image, (2, 0, 1)))
        mask = torch.tensor(mask)
            
        return image, mask, dist_map
    
    def __getitem__(self, idx):
        image, mask, dist_map = self.get_one_sample(idx)
        
        if random.random() < self.mixup:
            _idx = np.random.choice(range(0, self.__len__()))
            mixup_image, mixup_mask, dist_map = self.get_one_sample(_idx)
            lamb = np.random.beta(0.5, 0.5)
            image = lamb*image + (1-lamb)*mixup_image
            mask  = lamb*mask  + (1-lamb)*mixup_mask

        data = {"image": image, "mask": mask}
        if dist_map is not None:
            data["dist_map"] = dist_map
        return data
    
class RandomResize(DualTransform):
    """Resize the input to the given height and width.

    Args:
        height (int): desired height of the output.
        width (int): desired width of the output.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, height, width, scale_limit=0.0, scale_limit_x=None, scale_limit_y=None, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(RandomResize, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.scale_limit = to_tuple(scale_limit, bias=1.0)
        self.scale_limit_x = to_tuple(scale_limit_x, bias=1.0) if scale_limit_x is not None else None
        self.scale_limit_y = to_tuple(scale_limit_y, bias=1.0) if scale_limit_y is not None else None
        self.interpolation = interpolation
        
    def get_params(self):
        if self.scale_limit_x is not None and self.scale_limit_y is not None:
            scale_x = random.uniform(self.scale_limit_x[0], self.scale_limit_x[1])
            scale_y = random.uniform(self.scale_limit_y[0], self.scale_limit_y[1])
        else:
            scale_x = scale_y = random.uniform(self.scale_limit[0], self.scale_limit[1])
        return {"scale_x": scale_x, "scale_y": scale_y}

    def apply(self, img, scale_x=1.0, scale_y=1.0, **params):
        height = int(scale_y*self.height)
        width = int(scale_x*self.width)
        return F.resize(img, height=height, width=width, interpolation=self.interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        raise NotImplementedError

    def get_transform_init_args_names(self):
        return ("height", "width", "interpolation", "scale_limit", "scale_limit_x", "scale_limit_y")