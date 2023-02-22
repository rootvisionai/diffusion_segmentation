import torch.nn.functional
import torchvision.utils
import tqdm
import numpy as np

from .parts import *
from .lion import Lion

class SEQUNET(nn.Module):
    def __init__(
            self,
            input_channels=3,
            init_dim=64,
            dim=64,
            resnet_block_groups=8,
            dim_mults=(1, 2, 4, 8),
            steps=20,
            loss_function="MSELoss", # MSELoss | L1Loss
            learning_rate=0.001,
            optimizer="AdamW"
    ):
        super().__init__()
        """
        Extracted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
        """
        self.steps = steps
        # if dim=64 --->
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)] # ---> [64, 64,128,256,512]
        in_out = list(zip(dims[:-1], dims[1:]))               # ---> [(64, 64), (64, 128), (128, 256), (256, 512)]

        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out),
                block_klass(dim_out + dim_in, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        self.out_dim = input_channels

        self.final_res_block = block_klass(dim * 2, dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        self.loss_function = loss_function
        self.calculate_loss = getattr(nn, self.loss_function)(reduction="sum")
        self.optimizer = getattr(torch.optim, optimizer)(
            params=[{"params": self.parameters()}],
            lr=learning_rate
        ) if optimizer != "Lion" else Lion(
            params=[{"params": self.parameters()}],
            lr=learning_rate
        )

    def forward(self, x):

        x = self.init_conv(x)
        r = x.clone()

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            h.append(x)

            x = block2(x)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x)
        return torch.nn.functional.relu(self.final_conv(x))

    def autoencoder_training_step(self, x):
        out = self.forward(x)
        loss = self.calculate_loss(x, out)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def segmentation_training_step_sequential(self, x, m):
        step_values = torch.arange(1, 0 - 1 / self.steps, -1 / self.steps)[0:-1]

        for i, v in enumerate(step_values):
            if i < len(step_values) - 1:
                if i == 0:
                    x_ = x * v + m * (1 - v)
                # t_ = m * (1 - step_values[i + 1]) + x * step_values[i + 1]
                x_ = self.forward(x_.detach())  # image * 1 -> mask * 0.1 + image * 0.9

        loss = self.calculate_loss(x_, m)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def segmentation_training_step_partial(self, x, m):
        step_values = torch.arange(1, 0 - 1 / self.steps, -1 / self.steps)[0:-1]

        losses = []
        for i, v in enumerate(step_values):
            if i<len(step_values)-1:
                x_ = x*v + m*(1-v)
                t_ = m*(1-step_values[i+1]) + x*step_values[i+1]
                x_ = self.forward(x_) # image * 1 -> mask * 0.1 + image * 0.9
                loss = self.calculate_loss(x_, t_)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

        return np.mean(losses)

    def training_step(self, x, m,
                      sequential=True):

        if sequential:
            loss = self.segmentation_training_step_sequential(x, m)
        else:
            loss = self.segmentation_training_step_partial(x, m)

        return loss

    def infer(self, x, steps=None):
        for i in range(steps if steps else self.steps):
            with torch.no_grad():
                x = self.forward(x)
        return x

    def load_checkpoint(self, path, device="cuda"):
        ckpt_dict = torch.load(path, map_location=device)
        self.load_state_dict(ckpt_dict["model_state_dict"]) if "model_state_dict" in ckpt_dict else 0
        self.optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"]) if "optimizer_state_dict" in ckpt_dict else 0
        return ckpt_dict["last_epoch"]

    def save_checkpoint(self, path, epoch):
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "last_epoch": epoch
        }, path)
        return True



if __name__=="__main__":
    model = SEQUNET(
        input_channels=3,
        init_dim=64,
        dim=64,
        resnet_block_groups=8,
        dim_mults=(1, 2, 4, 8)
    )

    rand_tensor = torch.rand((2, 3, 512, 512))
    out = model(rand_tensor)
    print(out.shape)