import torch
from torch import nn

__all__ = ['MarkerBasedNorm']


class MarkerBasedNorm(nn.Module):
    def __init__(self, mean, std, in_channels, default_std=0.0001):
        super(MarkerBasedNorm, self).__init__()

        self.in_channles = in_channels
        self._default_std = default_std

        if mean is None or std is None:
            self.register_parameter('mean_by_channel', nn.Parameter(torch.zeros(1, self.in_channles, 1, 1)))
            self.register_parameter('std_by_channel', nn.Parameter(torch.ones(1, self.in_channles, 1, 1)))
        else:
            self.register_parameter('mean_by_channel', nn.Parameter(mean))
            self.register_parameter('std_by_channel', nn.Parameter(std))

    
    def forward(self, x):
        x = (x - self.mean_by_channel)/(self.std_by_channel + self._default_std)

        return x
