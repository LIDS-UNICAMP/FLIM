# noqa: D100

import logging

import torch.nn as nn

__all__ = ["LIDSConvNet"]


class LIDSConvNet(nn.Sequential):

    """A convolutional neural network.

    For now, it has only a feature extracor and no classifier.

    Attributes
    ----------
    feature_extractor : nn.Torch.Module
        A feature extractor formed by convolutional layers.

    """

    def __init__(self):
        """Initialize the class."""
        super(LIDSConvNet, self).__init__()
        self.feature_extractor = nn.Sequential()
        self.classifier = nn.Sequential()
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

        _y = self.feature_extractor(x)
        
        x = _y

        _y = self.classifier.forward(x)

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

        return self
