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


VOC_COLORMAP = [
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
    def _convert_to_segmentation_mask(mask):
        # This function converts a mask from the Pascal VOC format to the format required by AutoAlbument.
        #
        # Pascal VOC uses an RGB image to encode the segmentation mask for that image. RGB values of a pixel
        # encode the pixel's class.
        #
        # AutoAlbument requires a segmentation mask to be a NumPy array with the shape [height, width, num_classes].
        # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
        # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        return segmentation_mask

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index])
        mask[np.all(mask == (192, 224, 224), axis=-1)] = (0, 0, 0)
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