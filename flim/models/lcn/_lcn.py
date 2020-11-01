# noqa: D100

import logging
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

    def __init__(self, remove_boder=0):
        """Initialize the class."""
        super(LIDSConvNet, self).__init__()
        self.feature_extractor = nn.Sequential()
        # self.features = self.feature_extractor
        self.classifier = nn.Sequential()
        self._remove_border = remove_boder
        self._logger = logging.getLogger()

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

        for layer_name, layer in self.feature_extractor.named_children():
            _y = layer.forward(x)
            x = _y
        b = self._remove_border
        
        if b > 0:
            x = x[:,:, b:-b, b:-b]
            
        x = x.flatten(1)
        
        _y = self.classifier(x)

        return _y

    def to(self, device):
        """Move layer to ``device``.

        Move layer parameters to some specified device.

        Parameters
        ----------
        device : torch.device
            The device where to move.

        Returns
        -------
        Self
            The layer itself.


        Notes
        -----
        This method modifies the module in-place.

        """
        for _, layer in self.feature_extractor.named_children():
            layer.to(device)
        
        for _, layer in self.classifier.named_children():
            layer.to(device)

        return self


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


