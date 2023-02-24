import torch
import sys


class KNN(torch.nn.Module):
    def __init__(self, colors, classes, K=1):
        super().__init__()
        self.classes = torch.arange(0, len(classes))
        self.K = K

        if type(colors) == list:
            if type(colors[0]) == list or type(colors[0]) == tuple:
                colors = [torch.tensor(elm)/255 for elm in colors]
            else:
                pass
            colors = torch.stack(colors, dim=0).squeeze(1)
        elif type(colors) == torch.Tensor:
            pass
        else:
            print(f"WRONG INPUT TYPE {type(colors)}, must be one of {torch.Tensor} or {list} of {torch.Tensor}")
            sys.exit(1)
        a_norm = colors.norm(dim=-1, p=2)  # (batch size, 1)
        colors = colors.divide(a_norm.unsqueeze(1))
        self.colors = colors

    def preprocess(self, rgb_mask):
        self.shape = rgb_mask.shape
        rgb_mask = rgb_mask.permute(0, 2, 3, 1)
        rgb_mask = rgb_mask.reshape(rgb_mask.shape[0], rgb_mask.shape[1] * rgb_mask.shape[2],  rgb_mask.shape[3])
        return rgb_mask

    def forward(self, rgb_mask):
        rgb_mask = self.preprocess(rgb_mask)
        rgb_mask_ = torch.zeros_like(rgb_mask)

        for bs in range(rgb_mask.shape[0]):
            for pix in range(rgb_mask.shape[1]):
                rgb_mask_[bs, pix] = self._forward(rgb_mask[bs, pix].unsqueeze(0))

        rgb_mask_ = rgb_mask_.permute(0, 2, 1)
        return rgb_mask_.reshape(self.shape)

    def _forward(self, vector):
        if vector.sum()<=0:
            return torch.tensor([0, 0, 0])

        a_norm = vector.norm(dim=-1, p=2)
        vector = vector.divide(a_norm.unsqueeze(1))

        cos_sim = torch.nn.functional.linear(self.colors, vector).squeeze(1)

        cos_sim_topK = cos_sim.topk(1)
        indexes = cos_sim_topK[1][0].cpu()

        preds_int = self.classes[indexes]

        out = self.colors[preds_int]

        return out