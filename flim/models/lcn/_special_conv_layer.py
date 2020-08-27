# noqa: D100
import logging

import torch
from torch import nn
from torch.nn import Conv2d

import numpy as np

from skimage.util import view_as_windows, pad

from sklearn.cluster import MiniBatchKMeans


__all__ = ['SpecialConvLayer']

__operations__ = {
    "max_pool2d": nn.MaxPool2d,
    "relu": nn.ReLU,
    "batch_norm2d": nn.BatchNorm2d,
    "adap_avg_pool2d": nn.AdaptiveAvgPool2d,
}


class SpecialConvLayer(nn.Module):

    """Special convolutional layer trained from image markers.

    A SpecialConvLayer is a layer that extends ``torch.nn.Module`` \
    with the following operations:
    marker-based normalization, 2D-convolution, ReLU activation and 2D-pooling.
    It's kernels a learned from the image markers.

    Attributes
    ----------
    in_channels : int
        The number of input's channels.
    kernel_size: array_like:
        The kernel dimensions. If a single number :math:`k` if provided, \
        it will be interpreted as a kernel :math:`k \times k`.
    padding: array_like:
        The number of zeros to add to pad.
    bias : bool
        Whether to use bias or not.
    stride : int
         Stride for convolution.
    out_channels : int
        The number of ouput's channels.
    number_of_kernels_per_marker: int
        The number of kernel per marker.
    device : str
        Device where layer's parameters are stored.

    """

    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1,
                 bias=False,
                 number_of_kernels_per_marker=16,
                 activation_config=None,
                 pool_config=None,
                 device='cpu'):
        """Initialize the class.

        Parameters
        ----------
        in_channels : int
            The input channel number.
        kernel_size : int, optional
            The kernel dimensions. \
            If a single number :math:`k` if provided, it will be interpreted \
            as a kernel :math:`k \times k`, by default 3.
        padding : int, optional
            The number of zeros to add to pad, by default 1.
        stride : int, optional
            Stride for convolution, by default 1
        bias : bool, optional
            Whether to use bias or not., by default False
        number_of_kernels_per_marker : int, optional
            The number of kernel per marker., by default 16
        activation_config : [type], optional
            Activation options. See architecture speficitication format.
        pool_config : [type], optional
            Pooling options., by default None
        device : str, optional
            Device where layer's parameters are stored., by default 'cpu'.

        """
        super(SpecialConvLayer, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        self.out_channels = 0
        
        self._activation_config = activation_config
        self._pool_config = pool_config
        
        self.number_of_kernels_per_marker = number_of_kernels_per_marker

        self._conv = None
        self._activation = None
        self._pool = None

        self.device = device
        self._mean_by_channel = 0
        self._std_by_channel = 1
        
        self._logger = logging.getLogger()
        
    def initialize_weights(self, images, markers):
        """Learn kernel weights from image markers.

        Initialize layer with weights learned from image markers.

        Parameters
        ----------
        images : ndarray
            Array of images with shape :math:`(N, H, W, C)`.
        markers : list
            List of markers. For each image there is an ndarry with shape \
            :math:`3 \times N` where :math:`N` is the number of markers pixels.
            The first row is the markers pixels :math:`x`-coordinates, \
            second row is the markers pixels :math:`y`-coordinates, \
            and the third row is the markers pixel labels.

        """
        kernels_weights = self._calculate_weights(images, markers)
        self.out_channels = kernels_weights.shape[0]

        self._conv = Conv2d(self.in_channels,
                            kernels_weights.shape[0],
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            bias=self.bias,
                            padding=self.padding)

        self._conv.weight = nn.Parameter(
            torch.Tensor(np.rollaxis(kernels_weights, 3, 1)))

        self._conv.weight.requires_grad = False
        
        if self._activation_config is not None:
            self._activation = __operations__[
                self._activation_config['operation']](
                    **self._activation_config['params'])
        if self._pool_config is not None:
            self._pool = __operations__[self._pool_config['operation']](
                **self._pool_config['params'])
          
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
        self._mean_by_channel = self._mean_by_channel.to(device)
        self._std_by_channel = self._std_by_channel.to(device)
        
        self._conv = self._conv.to(device)

        if self._activation is not None:
            self._activation = self._activation.to(device)
        
        if self._pool is not None:
            self._pool = self._pool.to(device)
        
        return self
    
    def forward(self, x):
        """Apply special layer to an input tensor.

        Apply special layer to a batch images.

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
        self._logger.debug(
            "forwarding in special conv layer. Input shape %i", x.size())

        mean = self._mean_by_channel.view(1, -1, 1, 1)
        std = self._std_by_channel.view(1, -1, 1, 1)
        
        mean = mean.to(x.device)
        std = std.to(x.device)
        
        x = (x - mean)/std
        
        y = self._conv(x)

        if self._activation is not None:
            y = self._activation.forward(y)
        
        if self._pool is not None:
            # print("max pooling")
            y = self._pool.forward(y)
        
        return y
        
    def _calculate_weights(self, images, markers):
        """Calculate kernels weights from image markers.

        Parameters
        ----------
        images : ndarray
            Array of images with shape :math:`(N, H, W, C)`.
        markers : list
            List of markers. For each image there is an ndarry with shape \
            :math:`3 \times N` where :math:`N` is the number of markers pixels.
            The first row is the markers pixels :math:`x`-coordinates, \
            second row is the markers pixels :math:`y`-coordinates, \
            and the third row is the markers pixel labels.

        Returns
        -------
        ndarray
            Kernels weights in the shape \
            :math:`(N \times C \times H \times W)`.
        
        """
        patches, labels = self._generate_patches(images,
                                                 markers,
                                                 self.padding,
                                                 self.kernel_size,
                                                 self.in_channels)

        kernel_weights = _kmeans_roots(patches,
                                       labels,
                                       self.number_of_kernels_per_marker)
        
        norm = np.linalg.norm(
            kernel_weights.reshape(kernel_weights.shape[0], -1), axis=0)
        norm = norm.reshape(1, *kernel_weights.shape[1:])
        kernel_weights = kernel_weights/norm
        return kernel_weights
    
    def _generate_patches(self,
                          images,
                          markers,
                          padding=1,
                          kernel_size=3,
                          in_channels=3):
        """Get patches from markers pixels.

        Get a patch of size :math:`k \times k` around each markers pixel.
        
        ----------
        images : ndarray
            Array of images with shape :math:`(N, H, W, C)`.
        markers : list
            List of markers. For each image there is an ndarry with shape \
            :math:`3 \times N` where :math:`N` is the number of markers pixels.
            The first row is the markers pixels :math:`x`-coordinates, \
            second row is the markers pixels :math:`y`-coordinates, \
            and the third row is the markers pixel labels.
        in_channels : int
            The input channel number.
        kernel_size : int, optional
            The kernel dimensions. \
            If a single number :math:`k` if provided, it will be interpreted \
            as a kernel :math:`k \times k`, by default 3.
        padding : int, optional
            The number of zeros to add to pad, by default 1.

        Returns
        -------
        tuple[ndarray, ndarray]
            A array with all genereated pacthes and \
            an array with the label of each patch.
        
        """
        all_patches, all_labels = None, None
        for image, image_markers in zip(images, markers):
            if len(image_markers) == 0:
                continue
            image_pad = np.pad(image, ((padding, padding),
                                    (padding, padding), (0, 0)),
                            mode='constant', constant_values=0)
            
            patches = view_as_windows(image_pad,
                                      (kernel_size, kernel_size, in_channels),
                                      step=1)

            shape = patches.shape
            image_shape = image.shape
            
            markers_x = image_markers[0]
            markers_y = image_markers[1]
            labels = image_markers[2]
            
            mask = np.logical_and(
                markers_x < image_shape[0], markers_y < image_shape[1])
            
            markers_x = markers_x[mask]
            markers_y = markers_y[mask]
            labels = labels[mask]

            generated_patches = \
                patches[markers_x, markers_y].reshape(-1, *shape[3:])
            
            if all_patches is None:
                all_patches = generated_patches
                all_labels = labels
            else:
                all_patches = np.concatenate((all_patches, generated_patches))
                all_labels = np.concatenate((all_labels, labels))
        
        mean_by_channel = all_patches.mean(axis=(0, 1, 2), keepdims=True)
        std_by_channel = all_patches.std(axis=(0, 1, 2), keepdims=True)
        
        self._mean_by_channel = \
            torch.from_numpy(mean_by_channel).float().to(self.device)
        self._std_by_channel = \
            torch.from_numpy(std_by_channel).float().to(self.device)
        
        all_patches = (all_patches - mean_by_channel)/std_by_channel
        
        return all_patches, all_labels


def _kmeans_roots(patches,
                  labels,
                  n_clusters_per_label,
                  min_number_of_pacthes_per_label=16):
    """Cluster patch and return the root of each custer.

    Parameters
    ----------
    patches : ndarray
        Array of patches with shape :math:`((N, H, W, C))`
    labels : ndarray
        The label of each patch with shape :nath:`(N,)`
    n_clusters_per_label : int
        The number os clusters per label.
    min_number_of_pacthes_per_label : int, optional
        The mininum number of patches of a given label \
        for the clustering be performed , by default 16.

    Returns
    -------
    ndarray
        A array with all the roots.

    """
    roots = None
    min_number_of_pacthes_per_label = n_clusters_per_label

    possible_labels = np.unique(labels)

    for label in possible_labels:
        patches_of_label = patches[label == labels].astype(np.float32)
        # TODO get a value as arg.
        if patches_of_label.shape[0] > min_number_of_pacthes_per_label:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters_per_label, max_iter=100, random_state=42)
            kmeans.fit(patches_of_label.reshape(patches_of_label.shape[0], -1))
            
            roots_of_label = kmeans.cluster_centers_
        elif patches_of_label.shape[0] >= min_number_of_pacthes_per_label or \
                possible_labels.shape[0] == 1:
            roots_of_label = patches_of_label.reshape(
                patches_of_label.shape[0], -1)
        
        else:
            continue
        
        if roots is not None:
            roots = np.concatenate((roots, roots_of_label))
        else:
            roots = roots_of_label
    
    roots = roots.reshape(-1, *patches.shape[1:])
    return roots
