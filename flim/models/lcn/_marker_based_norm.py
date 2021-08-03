import torch
from torch import nn

__all__ = ['MarkerBasedNorm2d', 'MarkerBasedNorm3d']


class _MarkerBasedNorm(nn.Module):
    def __init__(self, in_channels, mean, std, epsilon=0.1):
        super(_MarkerBasedNorm, self).__init__()

        self.in_channles = in_channels
        self._epsilon = epsilon

        if mean is None:
            mean = torch.zeros(in_channels, dtype=torch.float32)

        if std is None:
            std = torch.ones(1, in_channels, dtype=torch.float32)

        self.register_buffer('mean_by_channel', nn.Parameter(mean))
        self.register_buffer('std_by_channel', nn.Parameter(std))

        self.weight = nn.Parameter(torch.ones(in_channels, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(in_channels, dtype=torch.float32))

    def forward(self, x):
        n_dims = x.dim() - 2
        correct_shape = (-1, *[1] * n_dims)
        x = (x - self.mean_by_channel.view(correct_shape))/(self.std_by_channel + self._epsilon).view(correct_shape)
        y = x * self.weight.view(correct_shape) + self.bias.view(correct_shape)
        return y

class MarkerBasedNorm2d(_MarkerBasedNorm):
    def __init__(self, in_channels, mean=None, std=None, *args, **kwargs):
        super().__init__(in_channels, mean, std, *args, **kwargs)


class MarkerBasedNorm3d(_MarkerBasedNorm):
    def __init__(self, in_channels, mean=None, std=None, *args, **kwargs):
        super().__init__(in_channels, mean, std, *args, **kwargs)
