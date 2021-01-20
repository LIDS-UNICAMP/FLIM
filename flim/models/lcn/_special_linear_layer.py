from inspect import Parameter
import logging

import numpy as np

import torch
from torch import nn

from sklearn.cluster import MiniBatchKMeans

import math

__all__ = ['SpecialConvLayer']

class SpecialLinearLayer(nn.Module):
    def __init__(self,
                in_features=None,
                out_features=None,
                bias=False,
                device='cpu'):
        super(SpecialLinearLayer, self).__init__()

        self.device = device

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        self.register_buffer('mean', torch.zeros(1, in_features))
        self.register_buffer('std', torch.ones(1, in_features))

        self._linear =  None

        self._logger = logging.getLogger()

    
    def initialize_weights(self,
                           features=None,
                           labels=None,
                           kernels_per_label=0.85):

        
        if features is not None and labels is not None:
            print("Calculaitn special linear layer weights")
            weights = self._calculate_weights(features, labels, kernels_per_label)
        else:
            assert self.in_features is not None, "in_features must me specified"
            assert self.out_features is not None, "out_features must me specified"
            
            weights = torch.rand(self.in_features,
                                         self.out_features).numpy().transpose()
        self.in_features = weights.shape[1]
        self.out_features = weights.shape[0]

  
        self._linear = nn.Linear(self.in_features, self.out_features, bias=self.bias)
        self._linear.weight = nn.Parameter(torch.Tensor(weights))
    
    def _calculate_weights(self, patches, labels, kernels_per_label=0.8):

        mean = patches.mean(axis=0, keepdims=True)
        std = patches.std(axis=0, keepdims=True) + 1e-6
        
        self.mean = \
            torch.from_numpy(mean).float().to(self.device)
        self.std = \
            torch.from_numpy(std).float().to(self.device)
        
        patches = (patches - mean)/std
        
        weights = _kmeans_roots(patches,
                                labels,
                                kernels_per_label)

        return weights

    def forward(self, x):
        """Apply special layer to an input tensor.

        Apply special layer to a batch images.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor in the shape :math:`(N, I)`.

        Returns
        -------
        torch.Tensor
            Output tensor with shape :math:`(N, O)`.


        """
        self._logger.debug(
            "forwarding in special linear layer. Input shape %i", x.size())

        mean = self.mean
        std = self.std

        if x.ndim == 3:
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)
            
        x = (x - mean)/std
    
        y = self._linear(x)
        
        return y

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
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        
        self._linear = self._linear.to(device)

        self.device = device

        return self

def _kmeans_roots(patches,
                  labels,
                  n_clusters_per_label):
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

    possible_labels = np.unique(labels)
    
    for label in possible_labels:
        patches_of_label = patches[label == labels].astype(np.float32)
        n_clusters = max(1, math.floor(patches_of_label.shape[0]*n_clusters_per_label))
        print("number of clusters", n_clusters)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, max_iter=100, random_state=42, init_size=max(2*n_clusters, 300))
        kmeans.fit(patches_of_label.reshape(patches_of_label.shape[0], -1))
        
        roots_of_label = kmeans.cluster_centers_

        if roots is not None:
            roots = np.concatenate((roots, roots_of_label))
        else:
            roots = roots_of_label
    
    roots = roots.reshape(-1, *patches.shape[1:])
    return roots