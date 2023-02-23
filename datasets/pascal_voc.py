"""
This code is from https://albumentations.ai/docs/autoalbument/examples/pascal_voc/
"""

import cv2
import numpy as np
import torch
from torchvision.datasets import VOCSegmentation

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


VOC_COLORMAP_OLD = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

VOC_COLORMAP_NEW = [
    (20, 0, 255),
    (0, 91, 255),
    (255, 0, 20),
    (255, 0, 138),
    (0, 205, 255),
    (29, 255, 0),
    (0, 136, 255),
    (255, 56, 0),
    (0, 255, 234),
    (0, 255, 210),
    (114, 255, 0),
    (126, 0, 255),
    (255, 0, 123),
    (0, 101, 255),
    (0, 255, 189),
    (0, 255, 45),
    (0, 255, 219),
    (60, 0, 255),
    (0, 255, 179),
    (0, 23, 255),
    (0, 141, 255),
]



class PascalVOC(VOCSegmentation):
    def __init__(
            self,
            transform=None,
            root="./datasets/pascal_voc",
            year="2012",
            image_set="train",
            download=False
    ):
        super().__init__(root=root, image_set=image_set, download=download, transform=transform, year=year)

    @staticmethod
    def _convert_to_new_labels(mask):
        for i, rgb_old in enumerate(VOC_COLORMAP_OLD):
            mask[np.all(mask == rgb_old, axis=-1)] = VOC_COLORMAP_NEW[i]
        mask[np.all(mask == VOC_COLORMAP_NEW[0], axis=-1)] = (0, 0, 0)
        mask[np.all(mask == (192, 224, 224), axis=-1)] = (0, 0, 0)
        return mask

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index])
        mask = self._convert_to_new_labels(mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"] / 255
            mask = transformed["mask"] / 255
            mask = mask.permute(2, 0, 1)
        return image, mask.to(torch.float32)

def get_dataloader(
        root="./datasets/pascal_voc",
        set_type="train",
        transform=None,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
):

    set = PascalVOC(
        transform=transform,
        root=root,
        year="2012",
        image_set=set_type,
        download=False
    )

    data_loader = DataLoader(
        set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return data_loader


def get_transforms(cfg, eval=True):
    train_transform = A.Compose(
        [
            A.Resize(cfg.input_size, cfg.input_size),

            A.ShiftScaleRotate(
                shift_limit=cfg.augmentations.ShiftScaleRotate.shift_limit,
                scale_limit=cfg.augmentations.ShiftScaleRotate.scale_limit,
                rotate_limit=cfg.augmentations.ShiftScaleRotate.rotate_limit,
                p=cfg.augmentations.ShiftScaleRotate.probability
            ),

            A.RGBShift(
                r_shift_limit=cfg.augmentations.RGBShift.r_shift_limit,
                g_shift_limit=cfg.augmentations.RGBShift.g_shift_limit,
                b_shift_limit=cfg.augmentations.RGBShift.b_shift_limit,
                p=cfg.augmentations.RGBShift.probability
            ),

            A.RandomBrightnessContrast(
                brightness_limit=cfg.augmentations.RandomBrightnessContrast.brightness_limit,
                contrast_limit=cfg.augmentations.RandomBrightnessContrast.contrast_limit,
                p=cfg.augmentations.RandomBrightnessContrast.probability
            ),
            ToTensorV2()
        ]
    )

    if eval:
        val_transform = A.Compose(
            [
                A.Resize(cfg.input_size, cfg.input_size),
                ToTensorV2()
            ]
        )

        return train_transform, val_transform

    else:
        return train_transform