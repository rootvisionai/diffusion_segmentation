import torch
import os
import tqdm
import torchvision
import numpy as np
import collections
from models.unet import SEQUNET
from datasets.pascal_voc import get_transforms, get_dataloader
from utils import load_config
from eval import visualize

if __name__=="__main__":
    cfg = load_config("./config.yml")

    # get dataloaders
    transform_tr, transform_ev = get_transforms(cfg.data)
    dl_tr = get_dataloader(
        root="./datasets/pascal_voc",
        set_type="train",
        transform=transform_tr,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True
    )
    dl_ev = get_dataloader(
        root="./datasets/pascal_voc",
        set_type="val",
        transform=transform_ev,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    # get model
    model = SEQUNET(
        input_channels=21,
        init_dim=64,
        dim=64,
        resnet_block_groups=8,
        dim_mults=(1, 2, 4, 8),
        steps=cfg.model.steps,
        loss_function="MSELoss",  # MSELoss | L1Loss
        learning_rate=cfg.training.learning_rate,
        optimizer="Lion",
        device=cfg.training.device,
        random_init_weights=torch.rand(21, 3, 1, 1).to(cfg.training.device)
    )
    model.to(cfg.training.device)
    # load checkpoint if exists
    checkpoint_path = f"init_dim[{cfg.model.init_dim}]_dim[{cfg.model.dim}]_resnet_block_groups[{cfg.model.resnet_block_groups}]_step[{cfg.model.steps}]_input[{cfg.data.input_size}]"
    checkpoint_path = os.path.join("checkpoints", checkpoint_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, "ckpt.pth")
    if os.path.isfile(checkpoint_path):
        last_epoch = model.load_checkpoint(checkpoint_path, device=cfg.training.device)
    else:
        last_epoch = 0

    model.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=2,
                                                           verbose=True,
                                                           threshold=0.01,
                                                           min_lr=0.0000001)

    # visualize(model, dl_ev, device=cfg.training.device)
    loss_hist = collections.deque(maxlen=30)
    for epoch in range(last_epoch, cfg.training.epochs):
        epoch_loss = []
        for i, (image, mask) in enumerate(dl_tr):
            for k in range(1):
                loss = model.training_step(image.to(cfg.training.device), mask.to(cfg.training.device), sequential=False)
                loss_hist.append(loss)
                print(f"EPOCH: {epoch} | ITER: {i}-{k}/{len(dl_tr)} | LOSS: {np.mean(loss_hist)} | LR: {model.optimizer.param_groups[0]['lr']}")

            if i % 10 == 0:
                model.eval()
                with torch.no_grad():
                    pred = model.infer(image.to(cfg.training.device), steps= cfg.model.steps).cpu()
                pred = torch.argmax(pred, dim=1).unsqueeze(1).cpu()/21
                torchvision.utils.save_image(pred, f"./inference_examples/pred{i}.png")
                torchvision.utils.save_image(image.cpu(), f"./inference_examples/image{i}.png")
                model.train()

            epoch_loss.append(loss)

        scheduler.step(np.mean(epoch_loss))
        model.save_checkpoint(path=checkpoint_path, epoch=epoch)
        # visualize(model, dl_ev, device=cfg.training.device)