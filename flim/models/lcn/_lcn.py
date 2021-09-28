# noqa: D100

import logging
import warnings
import torch

import torch.nn as nn
from torch.nn.modules.fold import Unfold

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
            warnings.warn("remove_border is deprecated and it will be removed.",
                DeprecationWarning,
                stacklevel=1)

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
        self._logger.info("doing forward")

        outputs = dict()
        if 'input' in self._outputs_to_save:
            outputs['input'] = x

        for layer_name, layer in self.named_modules():

            if layer_name in self._skips:
                concat = torch.cat([x, *[outputs[name] for name in self._skips[layer_name]]], axis=1)
                x = concat

            if isinstance(layer, (nn.Sequential, ParallelModule)):
                continue

            if isinstance(layer, nn.Fold):
                x = x.permute(0, 2, 1)

            y = layer(x)

            if isinstance(layer, nn.Unfold):
                y = y.permute(0, 2, 1)

            if layer_name in self._outputs_to_save:
                outputs[layer_name] = y

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

        return torch.cat(outputs, 0)


