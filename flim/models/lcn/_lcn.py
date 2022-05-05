# noqa: D100

import logging
import warnings

import torch
import torch.nn as nn

__all__ = ["LIDSConvNet", "ParallelModule"]


class LIDSConvNet(nn.Sequential):

    """A convolutional neural network.

    For now, it has only a feature extracor and no classifier.

    Attributes
    ----------
    feature_extractor : nn.Torch.Module
        A feature extractor formed by convolutional layers.

    classifier : nn.Torch.Module
        A classifier.

    """

    def __init__(self, remove_boder=None, skips=None, outputs_to_save=None):
        """Initialize the class."""
        super(LIDSConvNet, self).__init__()
        self._logger = logging.getLogger()

        if remove_boder:
            warnings.warn(
                "remove_border is deprecated and it will be removed.",
                DeprecationWarning,
                stacklevel=1,
            )

        self._skips = skips
        self._outputs_to_save = outputs_to_save

    def forward(self, x):
        """Apply the network to an input tensor.

        Apply the network to a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor in the shape :math:`(N, C, H, W)`.

        Returns
        -------
        torch.Tensor
            Output tensor with shape :math:`(N, C^\prime,  H, W)`.
            The output height and width depends on the paramenters \
            used to define the layer.

        """

        for _, layer in self.named_children():

            if isinstance(layer, nn.Fold):
                x = x.permute(0, 2, 1)

            y = layer(x)

            if isinstance(layer, nn.Unfold):
                y = y.permute(0, 2, 1)

            x = y

        y = x

        return y


class ParallelModule(nn.ModuleList):

    """A module where every sub module is applied in parallel.

    Each sub module is applied in parallel and the output \
         is concatenated.

    """

    def __init__(self):
        super(ParallelModule, self).__init__()

    def forward(self, x):
        """Apply the module to an input tensor.

        Apply the module to a batch of images.\
        Each sub module is aplied in parallel and \
        the outputs are concatenated.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor in the shape :math:`(N, C, H, W)`.

        Returns
        -------
        torch.Tensor
            Output tensor with shape :math:`(N, C^\prime,  H, W)`.
            The output height and width depends on the paramenters \
            used to define the layer.

        """

        outputs = []

        for module in self.children():
            outputs.append(module(x))

        return torch.cat(outputs, 1)
