from pathlib import Path
from typing import List

import cv2
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from einops import rearrange
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset


class Skin_Leison(Dataset):
    """Some Information about Skin_Leison"""

    def __init__(self, data_path: str, transforms: List = []):
        super(Skin_Leison, self).__init__()
        self.img_list = sorted((Path(data_path)/'image').rglob('*.jpg'))
        self.label_list = sorted((Path(data_path)/'label').rglob('*.jpg'))

        self.transforms = A.Compose([
            *transforms,
            A.transforms.Normalize(mean=0, std=1, max_pixel_value=255),
            ToTensorV2()
        ])

    def __getitem__(self, index):
        img = cv2.imread(str(self.img_list[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.imread(str(self.label_list[index]), cv2.IMREAD_GRAYSCALE)
        aug = self.transforms(image=img, mask=label)
        return aug['image'], torch.round(aug['mask']/255).unsqueeze(0)

    def __len__(self):
        return len(self.img_list)

    def visualize(self, index):
        plt.rcParams.update({'figure.max_open_warning': 0})
        img1, img2 = self[index]
        fig, axes = plt.subplots(ncols=2, figsize=(18, 10))
        for col, image in zip(axes, [img1, img2]):
            if image.ndim == 3:
                image = rearrange(image, 'c h w -> h w c')
            col.imshow(image, cmap='gray')
        fig.tight_layout()
