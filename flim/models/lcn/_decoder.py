import torch
import torch.nn as nn

from ...utils import compute_importance

__all__ = ["Decoder"]


class Decoder(nn.Module):
    def __init__(self, images, markers, n_classes, device="cpu"):

        super(Decoder, self).__init__()

        self.n_classes = n_classes
        self.device = device

        self.register_buffer(
            "importance_by_channel",
            torch.from_numpy(compute_importance(images, markers, n_classes)).float(),
        )

    def forward(self, X):
        y = X.unsqueeze(1).repeat(
            1, self.n_classes, 1, 1, 1
        ) * self.importance_by_channel.view(1, self.n_classes, -1, 1, 1)

        comb = torch.sum(y, axis=2)
        comb[comb < 0] = 0

        return comb
