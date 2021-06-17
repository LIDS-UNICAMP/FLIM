# noqa: D100
import logging

import torch
from torch import nn
from torch.nn import Conv2d

import numpy as np

from skimage.util import view_as_windows, pad

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA

from scipy.spatial import distance

import math


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
    dilation : int
         Dilaton for atrous convolution.
    out_channels : int
        The number of ouput's channels.
    number_of_kernels_per_marker: int
        The number of kernel per marker.
    device : str
        Device where layer's parameters are stored.

    """

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 kernel_size=3,
                 padding=1,
                 stride=1,
                 bias=False,
                 dilation=1,
                 number_of_kernels_per_marker=16,
                 use_random_kernels=False,
                 use_pca=False,
                 activation_config=None,
                 pool_config=None,
                 zero_border=False,
                 normalize_before=True,
                 default_std=1e-6,
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
            Stride for convolution, by default 1.
        dilation : int
            Dilaton for atrous convolution.
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

        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = [padding, padding]
        else:
            self.padding = padding

        self.stride = stride

        if isinstance(dilation, int):
            self.dilation = [dilation, dilation]
        else:
            self.dilation = dilation

        self.bias = bias
        self.out_channels = out_channels

        self._activation_config = activation_config
        self._pool_config = pool_config
        self.number_of_kernels_per_marker = number_of_kernels_per_marker

        self._conv = None
        self._activation = None
        self._pool = None
        
        self._use_random_kernels = use_random_kernels
        self._use_pca = use_pca

        self._zero_border = zero_border

        self._normalize_before = normalize_before

        #self.register_buffer('mean_by_channel', torch.zeros(1, 1, 1, self.in_channels))
        #self.register_buffer('std_by_channel', torch.ones(1, 1, 1, self.in_channels))
        
        #self.mean_by_channel = nn.Parameter(torch.zeros(1, 1, 1, self.in_channels))
        #self.std_by_channel = nn.Parameter(torch.zeros(1, 1, 1, self.in_channels))
        
        num_channels = in_channels if normalize_before else out_channels

        self.register_parameter('mean_by_channel', nn.Parameter(torch.zeros(1, num_channels, 1, 1)))
        self.register_parameter('std_by_channel', nn.Parameter(torch.ones(1, num_channels, 1, 1)))
        
        self.device = device

        self._default_std = default_std
        
        self._logger = logging.getLogger()
        
    def initialize_weights(self, images=None, markers=None, kernels_number=1):
        """Learn kernel weights from image markers.

        Initialize layer with weights learned from image markers,
        or with random kernels if no image and markers are passed.

        Parameters
        ----------
        images : ndarray
            Array of images with shape :math:`(N, H, W, C)`,
            by default None.
        markers : ndarray
            A set of image markes as label images with size :math:`(N, H, W)`.\
            The label 0 denote no label, by default None.
        kernels_number: int
            If no images and markers are passed, the number of kernels must \
            must me specified. If images and markers are not None, \
            this argument is ignored. 

        """
        if self._use_random_kernels:
            kernels_weights = _create_random_pca_kernels(n=self.out_channels * 10,
                                                         k=self.out_channels,
                                                         in_channels=self.in_channels,
                                                         kernel_size=self.kernel_size)
            
        elif images is not None and markers is not None:
            kernels_weights = self._calculate_weights(images, markers)
            kernels_weights = np.rollaxis(kernels_weights, 3, 1)

            if  self.out_channels is not None and self.out_channels < kernels_weights.shape[0] and np.prod(kernels_weights.shape[1:]) > self.out_channels:
                kernels_weights = _select_kernels_with_pca(kernels_weights, self.out_channels)
        
            elif self.out_channels is not None and self.out_channels < kernels_weights.shape[0]:
                kernels_weights = _kmeans_roots(kernels_weights, np.ones(kernels_weights.shape[0]), self.out_channels)

        else:
            kernels_weights = torch.rand(kernels_number,
                                         self.in_channels,
                                         self.kernel_size[0],
                                         self.kernel_size[1]).numpy()

        self.out_channels = kernels_weights.shape[0]
        
        # print(self.std_by_channel)

        _conv = Conv2d(self.in_channels,
                            kernels_weights.shape[0],
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            bias=self.bias,
                            padding=self.padding,
                            dilation=self.dilation)

        _conv.weight = nn.Parameter(torch.Tensor(kernels_weights))

        _conv.weight.requires_grad = False
        
        
        if images is None or markers is None:
            print("Initialing with xavier")
            n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
            _conv.weight.data.normal_(0, math.sqrt(2. / n))
            
            if _conv.bias is not None:
                _conv.bias.data.zero_()
                
        self.add_module("__conv__", _conv)

        if not self._normalize_before:
            self._compute_mean_and_std_after_conv(images, markers)
        
        if self._activation_config is not None:
            _activation = __operations__[
                self._activation_config['operation']](
                    **self._activation_config['params'])
                
            self.add_module("activation", _activation)
                
        if self._pool_config is not None:
            _pool = __operations__[self._pool_config['operation']](
                **self._pool_config['params'])
            
            self.add_module("pool", _pool)

    def update_weights(self, images, old_markers, new_markers):
        """Learn kernel weights from image markers.

        Initialize layer with weights learned from image markers.

        Parameters
        ----------
        images : ndarray
            Array of images with shape :math:`(N, H, W, C)`.
        old_markers : ndarray
            A set of image markes as label images with size :math:`(N, H, W)`.\
            The label 0 denote no label.
        new_markers : ndarray
            A set of image markes as label images with size :math:`(N, H, W)`.\
            The label 0 denote no label.

        """
        assert images is not None and old_markers is not None \
            and new_markers is not None, \
            "Images, old marker and new markers must be provided"
        
        assert images.shape[:-1] == old_markers.shape and \
            images.shape[:-1] == new_markers.shape, \
            "Images and markers must have compatible shapes"

        use_all_patches = self.in_channels != images.shape[-1]

        self.in_channels = images.shape[-1]

        old_markers_patches, old_markers_labels = self._generate_patches(
            images, old_markers, self.padding,
            self.kernel_size, self.in_channels)

        patches, labels = self._generate_patches(images,
                                                 new_markers,
                                                 self.padding,
                                                 self.kernel_size,
                                                 self.in_channels)

        all_patches = np.concatenate((old_markers_patches, patches))
        all_labels = np.concatenate((old_markers_labels, labels))

        mean_by_channel = all_patches.mean(axis=(0, 1, 2), keepdims=True)
        std_by_channel = all_patches.std(axis=(0, 1, 2), keepdims=True)
        
        self.mean_by_channel = \
            torch.from_numpy(mean_by_channel).float().to(self.device)
        self.std_by_channel = \
            torch.from_numpy(std_by_channel).float().to(self.device)
 
        if use_all_patches:
            patches = all_patches
            labels = all_labels
        
        patches = (patches - mean_by_channel)/std_by_channel
        
        kernel_weights = _kmeans_roots(patches,
                                       labels,
                                       self.number_of_kernels_per_marker)
        
        kernels_shape = kernel_weights.shape
        kernel_weights = kernel_weights.reshape(kernels_shape[0], -1)
        norm = np.linalg.norm(kernel_weights, axis=1)
        norm = np.expand_dims(norm, 1)
        kernel_weights = kernel_weights/norm
        kernel_weights = kernel_weights.reshape(kernels_shape)

        if not use_all_patches:
            old_weights = self._conv.weight.detach().permute(0, 2, 3, 1).cpu().numpy()

            kernel_weights = np.concatenate((old_weights, kernel_weights))

        self.out_channels = kernel_weights.shape[0]

        self._conv = Conv2d(self.in_channels,
                            kernel_weights.shape[0],
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            bias=self.bias,
                            padding=self.padding)

        self._conv.weight = nn.Parameter(
            torch.Tensor(np.rollaxis(kernel_weights, 3, 1)))

        self._conv.to(self.device)

        self._conv.weight.requires_grad = False

    def remove_similar_filters(self, similarity_level=0.85):
        """Remove redundant filters.

        Remove redundant filter bases on inner product.

        Parameters
        ----------
        similarity_level : float, optional
            A value in range :math:`(0, 1]`. \
            If filters have inner product greater than value, \
            only one of them are kept. by default 0.85.
            All filters in this layer has euclidean norm equal to 1.

        """
        assert 0 < similarity_level <= 1,\
            "Similarity must be in range (0, 1]"

        filters = self._conv.weight

        similarity_matrix = _compute_similarity_matrix(filters)

        keep_filter = np.full(filters.size(0), True, np.bool)

        for i in range(0, filters.size(0)):
            if keep_filter[i]:

                mask = similarity_matrix[i] >= similarity_level
                indices = np.where(mask)

                keep_filter[indices] = False
        
        selected_filters = filters[keep_filter]

        self.out_channels = selected_filters.size(0)

        self._conv = Conv2d(self.in_channels,
                            selected_filters.size(0),
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            bias=self.bias,
                            padding=self.padding)

        self._conv.weight = nn.Parameter(selected_filters)

        self._conv.to(self.device)

        # self._conv.weight.requires_grad = False

    def remove_filters(self, filter_indices):
        """Remove Conv2D filters.

        Following layers must be updated.

        Parameters
        ----------
        filter_indices : ndarray
            A 1D array with indices to remove.
        """        
        assert filter_indices is not None, "filter indices must be provided"

        current_kernels = self._conv.weight.detach()

        mask = np.in1d(np.arange(current_kernels.size(0)), filter_indices)

        mask = np.logical_not(mask)

        selected_kernels = current_kernels[mask]

        self.out_channels = selected_kernels.size(0)

        self._conv = Conv2d(self.in_channels,
                            selected_kernels.size(0),
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            dilation=self.dilation,
                            bias=self.bias,
                            padding=self.padding)

        self._conv.weight = nn.Parameter(selected_kernels)

        self._conv.to(self.device)

        self._conv.weight.requires_grad = False

      
    '''def to(self, device):
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
        super(SpecialConvLayer, self).to()
        
        self.mean_by_channel.weight = self.mean_by_channel.to(device)
        self.std_by_channel.weight = self.std_by_channel.to(device)
    
        print("parameter weight", self.mean_by_channel.weight.device)
        print("parameter", self.mean_by_channel.device)
        print("device", device)
        
        self._conv = self._conv.to(device)

        self.device = device

        if self._activation is not None:
            self._activation = self._activation.to(device)
        
        if self._pool is not None:
            self._pool = self._pool.to(device)
        
        return self'''
    
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

        if self._normalize_before:
            x = (x - self.mean_by_channel)/(self.std_by_channel + self._default_std)
        
        for _, layer in self.named_children():
            x = layer.forward(x)
            
        y = x

        if not self._normalize_before:
            y = (y - self.mean_by_channel)/(self.std_by_channel + self._default_std)

        if self._zero_border:
            y[:, :, 0:self.padding[0], 0:self.padding[0]] = 0
            y[:, :, -self.padding[1]:, -self.padding[1]:] = 0

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
        updating : bool

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

        if self._normalize_before:
            mean_by_channel = patches.mean(axis=(0, 1, 2), keepdims=True)
            std_by_channel = patches.std(axis=(0, 1, 2), keepdims=True)
            
            self.mean_by_channel.data = torch.from_numpy(mean_by_channel).view(1, -1, 1, 1).float()
            self.std_by_channel.data = torch.from_numpy(std_by_channel).view(1, -1, 1, 1).float()
                    
            # print(self.std_by_channel)
            
            patches = (patches - mean_by_channel)/std_by_channel

        kernel_weights = _kmeans_roots(patches,
                                       labels,
                                       self.number_of_kernels_per_marker)

        kernels_shape = kernel_weights.shape
        kernel_weights = kernel_weights.reshape(kernels_shape[0], -1)
        norm = np.linalg.norm(kernel_weights, axis=1)
        norm = np.expand_dims(norm, 1)
        #print(norm)
        kernel_weights = kernel_weights/(norm + 0.00001)
        kernel_weights = kernel_weights.reshape(kernels_shape)
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

            kernel_size = np.array(kernel_size)
            dilation = np.array(self.dilation)

            dilated_kernel_size = kernel_size + (dilation - 1) * (kernel_size-1)
            dilated_padding = dilated_kernel_size // 2
            image_pad = np.pad(image, ((dilated_padding[0], dilated_padding[0]),
                                    (dilated_padding[1], dilated_padding[1]), (0, 0)),
                            mode='constant', constant_values=0)

            patches = view_as_windows(image_pad,
                                      (dilated_kernel_size[0], dilated_kernel_size[1], in_channels),
                                      step=1)
                                      
            if self.dilation[0] > 1 or self.dilation[1] > 0:
                r = np.arange(0, dilated_kernel_size[0], self.dilation[0])
                s = np.arange(0, dilated_kernel_size[1], self.dilation[1])
                patches = patches[:, :, :, r, : , :][:, :, :, :, r , :]

            shape = patches.shape
            image_shape = image.shape

            indices = np.where(image_markers != 0)
            markers_x = indices[0]
            markers_y = indices[1]
            labels = image_markers[indices] - 1
            
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
        
        return all_patches, all_labels

    def _compute_mean_and_std_after_conv(self, images, markers):
        device = self.device

        if images is None or markers is None:
            return

        torch_images = torch.from_numpy(images).float().permute(0, 3, 1, 2).to(device)

        conv_out = self.conv(torch_images).detach().cpu().permute(0, 2, 3, 1).numpy()

        patches, _ = self._generate_patches(conv_out,
                                        markers,
                                        self.padding,
                                        self.kernel_size,
                                        conv_out.shape[-1])

        
        mean_by_channel = patches.mean(axis=(0, 1, 2), keepdims=True)
        std_by_channel = patches.std(axis=(0, 1, 2), keepdims=True)
        
        self.mean_by_channel.data = torch.from_numpy(mean_by_channel).view(1, -1, 1, 1).float()
        self.std_by_channel.data = torch.from_numpy(std_by_channel).view(1, -1, 1, 1).float()


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
    print("Number of patches", patches.shape[0])
    for label in possible_labels:
        patches_of_label = patches[label == labels].astype(np.float32)
        # TODO get a value as arg.
        if patches_of_label.shape[0] > min_number_of_pacthes_per_label:
            # TODO remove fix random_state
            #kmeans = MiniBatchKMeans(
            #    n_clusters=n_clusters_per_label, max_iter=300, random_state=42, init_size=3 * n_clusters_per_label)

            kmeans = KMeans(n_clusters=n_clusters_per_label, max_iter=100, tol=0.001)
            kmeans.fit(patches_of_label.reshape(patches_of_label.shape[0], -1))
            
            roots_of_label = kmeans.cluster_centers_
        elif patches_of_label.shape[0] >= min_number_of_pacthes_per_label or \
             roots is None:
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

def _compute_similarity_matrix(filters):
    """Compute similarity matrix.

    The similarity between two filter is the inner product between them.

    Parameters
    ----------
    filters : torch.Tensor
        Array of filters.

    Returns
    -------
    ndarray
        A matrix N x N.

    """
    assert filters is not None, "Filter must be provided"

    _filters = filters.detach().flatten(1).cpu().numpy()

    similiraty_matrix = distance.pdist(_filters, metric=np.inner)

    return distance.squareform(similiraty_matrix)


def _create_random_kernels(n ,in_channels, kernel_size):
    kernels = np.random.rand(n, in_channels, *kernel_size)

    return kernels

def _enforce_norm(kernels):
    kernels_shape = kernels.shape
    flattened_kernels = kernels.reshape(kernels_shape[0], -1)

    norm = np.linalg.norm(flattened_kernels, axis=1, keepdims=True)

    normalized = flattened_kernels/norm

    mean = normalized.mean(axis=1, keepdims=True)

    centered = normalized - mean

    centered = centered.reshape(*kernels_shape)

    return centered

def _create_random_pca_kernels(n, k, in_channels, kernel_size):

    if isinstance(kernel_size, int):
      kernel_size = [kernel_size]*2
    elif isinstance(kernel_size, list) and len(kernel_size) == 1:
      kernel_size = kernel_size*2

    kernels = _enforce_norm(_create_random_kernels(n, in_channels, kernel_size))

    kernels_pca = _select_kernels_with_pca(kernels, k)

    return kernels_pca

def _select_kernels_with_pca(kernels, k):
    kernels_shape = kernels.shape

    kernels_flatted = kernels.reshape(kernels_shape[0], -1)
    if k > kernels_flatted.shape[0] or k > kernels_flatted.shape[1]:
        k = min(kernels_flatted.shape[0], kernels_flatted.shape[1])

    pca = PCA(n_components=k)
    pca.fit(kernels_flatted)

    kernels_pca = pca.components_

    kernels_pca = kernels_pca.reshape(-1, *kernels_shape[1:])

    return kernels_pca
