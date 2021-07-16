import torch
from torch import nn

__all__ = ['MarkerBasedNorm2d', 'MarkerBasedNorm3d']


class _MarkerBasedNorm(nn.Module):
    def __init__(self, in_channels, mean, std, default_std=0.0001):
        super(_MarkerBasedNorm, self).__init__()

        self.in_channles = in_channels
        self._default_std = default_std

        self.register_parameter('mean_by_channel', nn.Parameter(mean))
        self.register_parameter('std_by_channel', nn.Parameter(std))

    
    def forward(self, x):
        x = (x - self.mean_by_channel)/(self.std_by_channel + self._default_std)

        return x

class MarkerBasedNorm2d(_MarkerBasedNorm):
    def __init__(self, in_channels, mean=None, std=None, default_std=0.0001):
        if mean is None:
            mean = torch.zeros(1, in_channels, 1, 1, dtype=torch.float32)

        if std is None:
            std = torch.ones(1, in_channels, 1, 1, dtype=torch.float32)

        super().__init__(in_channels, mean, std, default_std)


class MarkerBasedNorm3d(_MarkerBasedNorm):
    def __init__(self, in_channels, mean=None, std=None, default_std=0.0001):
        if mean is None:
            mean = torch.zeros(1, in_channels, 1, 1, 1, dtype=torch.float32)

        if std is None:
            std = torch.ones(1, in_channels, 1, 1, 1, dtype=torch.float32)

        super().__init__(in_channels, mean, std, default_std)
