import torch
from torch import nn
from torch.nn import Conv2d

import numpy as np

from skimage.util import view_as_windows, pad

from sklearn.cluster import MiniBatchKMeans, KMeans

import logging

__operations__ = {
    "max_pool2d": nn.MaxPool2d,
    "relu": nn.ReLU,
    "linear": nn.Linear,
    "batch_norm2d": nn.BatchNorm2d,
    "dropout": nn.Dropout,
    "adap_avg_pool2d": nn.AdaptiveAvgPool2d,
}

class SpecialConvLayer(nn.Module):
    def __init__(self, in_channels, kernel_size=3, padding=1, stride=1, bias=False, device='cpu', kmeans_number_of_kernels=None, activation_config=None, pool_config=None):
        super(SpecialConvLayer, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        self.out_channels = 0
        
        self.activation_config = activation_config
        self.pool_config = pool_config
        
        if kmeans_number_of_kernels is None:
            self.kmeans_number_of_kernels = 16
        else:
            self.kmeans_number_of_kernels = kmeans_number_of_kernels

        self.conv = None
        self.activation = None
        self.pool = None

        self.device = device
        self.mean_by_channel = 0
        self.std_by_channel = 1
        
        self.kernels_of_label = []

        self.logger = logging.getLogger()
        
    def initialize_weights(self, images, markers):
        kernels_weights = self._calculate_weights(images, markers)
        self.out_channels = kernels_weights.shape[0]
        self.conv = Conv2d(self.in_channels, kernels_weights.shape[0], kernel_size=self.kernel_size, stride=self.stride, bias=self.bias, padding=self.padding)
        self.conv.weight = nn.Parameter(torch.Tensor(np.rollaxis(kernels_weights, 3, 1)))
        self.conv.weight.requires_grad = False
        #self.conv = self.conv.to(self.device)
        
        #print(self.pool_config['params'])
        if self.activation_config is not None:
            self.activation = __operations__[self.activation_config['operation']](**self.activation_config['params'])
        if self.pool_config is not None:
            self.pool = __operations__[self.pool_config['operation']](**self.pool_config['params'])
        
        #print(self.activation)
        
    def to(self, device):
        #print("moving to ", device)
        self.mean_by_channel = self.mean_by_channel.to(device)
        self.std_by_channel = self.std_by_channel.to(device)
        
        self.conv = self.conv.to(device)

        if self.activation is not None:
            self.activation = self.activation.to(device)
        
        if self.pool is not None:
            self.pool = self.pool.to(device)
        
        return self
        
    def forward(self, x):
        self.logger.debug("forwarding in special conv layer. Input shape {}".format(x.size()))
        mean = self.mean_by_channel.view(1, -1, 1, 1)
        std = self.std_by_channel.view(1, -1, 1, 1)
        
        mean = mean.to(x.device)
        std = std.to(x.device)
        
        x = (x - mean)/std
        
        y = self.conv(x)

        if self.activation is not None:
            y = self.activation.forward(y)
        
        if self.pool is not None:
            #print("max pooling")
            y = self.pool.forward(y)
        
        return y
        
    def _calculate_weights(self, images, markers):
        patches, labels = self._generate_patches(images, markers, self.padding, self.kernel_size, self.in_channels)
        kernel_weights = self._kmeans_roots(patches, labels, self.kmeans_number_of_kernels)
        
        norm = np.linalg.norm(kernel_weights.reshape(kernel_weights.shape[0], -1), axis=0)
        norm = norm.reshape(1, *kernel_weights.shape[1:])
        kernel_weights = kernel_weights/norm
        return kernel_weights
    
    def _generate_patches(self, images, markers, padding=1, kernel_size=3, in_channels=3):
        all_patches, all_labels = None, None
        for image, image_markers in zip(images, markers):
            if len(image_markers) == 0:
                continue
            image_pad = pad(image, ((padding, padding),
                                    (padding, padding), (0, 0)), mode='constant')
            
            patches = view_as_windows(image_pad, (kernel_size, kernel_size, in_channels), step=1)

            shape = patches.shape
            image_shape = image.shape
            
            markers_x = image_markers[0]
            markers_y = image_markers[1]
            labels = image_markers[2]
            
            mask = np.logical_and(markers_x < image_shape[0], markers_y < image_shape[1])
            
            markers_x = markers_x[mask]
            markers_y = markers_y[mask]
            labels = labels[mask]
            
            #print(markers)

            generated_patches = patches[markers_x, markers_y].reshape(-1, *shape[3:])
            
            if all_patches is None:
                all_patches = generated_patches
                all_labels = labels
            else:
                all_patches = np.concatenate((all_patches, generated_patches))
                all_labels = np.concatenate((all_labels, labels))
        
        mean_by_channel = all_patches.mean(axis=(0, 1, 2), keepdims=True)
        std_by_channel = all_patches.std(axis=(0, 1, 2), keepdims=True)
        
        #mean_by_channel = images.mean(axis=(0, 1, 2), keepdims=True)
        #std_by_channel = images.std(axis=(0, 1, 2), keepdims=True)
        
        self.mean_by_channel = torch.from_numpy(mean_by_channel).float().to(self.device)
        self.std_by_channel = torch.from_numpy(std_by_channel).float().to(self.device)
        
        all_patches = (all_patches - mean_by_channel)/std_by_channel
        
        #((labels == 1).sum())
        
        return all_patches, all_labels
    
    def _kmeans_roots(self, patches, labels, n_clusters_per_label, min_number_of_pacthes_per_label=16):
        roots = None
        #min_number_of_pacthes_per_label = n_clusters_per_label
        num_classes = labels.max()
        possible_labels = np.unique(labels)
        for label in possible_labels:
            patches_of_label = patches[label == labels].astype(np.float32)
            #TODO get a value as an arg?
            if patches_of_label.shape[0] > n_clusters_per_label:
                kmeans = MiniBatchKMeans(n_clusters=n_clusters_per_label, max_iter=100, random_state = 42)
                kmeans.fit(patches_of_label.reshape(patches_of_label.shape[0], -1))
                
                roots_of_label = kmeans.cluster_centers_
            elif patches_of_label.shape[0] >= min_number_of_pacthes_per_label or possible_labels.shape[0] == 1:
                roots_of_label = patches_of_label.reshape(patches_of_label.shape[0], -1)
            
            else:
                continue
            #print("There are {} roots of label {}".format(roots_of_label.shape[0], label))
            
            self.kernels_of_label.append(roots_of_label.shape[0])
            
            if roots is not None:
                roots = np.concatenate((roots, roots_of_label))
            else:
                roots = roots_of_label
        
        roots = roots.reshape(-1, *patches.shape[1:])
        return roots