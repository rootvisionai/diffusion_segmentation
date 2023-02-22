from models.unet import SEQUNET
from datasets.pascal_voc import get_transforms
from utils import load_config

import os
import torch
import torchvision
import glob
import cv2

def run(cfg, path_to_images, steps=None):
    # get model
    model = SEQUNET(
        input_channels=3,
        init_dim=64,
        dim=64,
        resnet_block_groups=8,
        dim_mults=(1, 2, 4, 8),
        steps=cfg.model.steps,
        loss_function="MSELoss",  # MSELoss | L1Loss
        learning_rate=cfg.training.learning_rate,
        optimizer="Lion"
    )
    model.to(cfg.training.device)

    # load checkpoint if exists
    checkpoint_path = f"init_dim[{cfg.model.init_dim}]_dim[{cfg.model.dim}]_resnet_block_groups[{cfg.model.resnet_block_groups}]_step[{cfg.model.steps}]_input[{cfg.data.input_size}]"
    checkpoint_path = os.path.join("checkpoints", checkpoint_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, "ckpt.pth")
    if os.path.isfile(checkpoint_path):
        model.load_checkpoint(checkpoint_path, device=cfg.training.device)
    model.to(cfg.training.device)

    # get images
    _, transform = get_transforms(cfg.data, eval=True)
    image_paths = glob.glob(os.path.join(path_to_images, "*.jpg"))
    print("INFERENCE IMAGES:")
    for p in image_paths:
        print(f"---> {p}")
    images = [cv2.imread(p) for p in image_paths]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    images = [transform(image=img)["image"]/255 for img in images]
    images = torch.stack(images, dim=0)

    model.eval()
    with torch.no_grad():
        preds = model.infer(images.to(cfg.training.device), steps=steps if steps else cfg.model.steps).cpu()

    for i, path in enumerate(image_paths):
        torchvision.utils.save_image(preds[i].unsqueeze(0).cpu(), path.replace(".jpg", "_pred.jpg"))

if __name__ == "__main__":
    cfg = load_config("./config.yml")
    run(cfg=cfg, steps=50, path_to_images="test_images")