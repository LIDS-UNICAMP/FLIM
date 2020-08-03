import torch
import torch.nn as nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

import numpy as np

from skimage.util import view_as_windows, pad, view_as_blocks
from skimage import io
from skimage.color import rgb2lab, lab2rgb

from scipy.ndimage import label


import os.path as path
import os

import pyift.pyift as ift

import math

from .special_conv_layer import SpecialConvLayer
from .LCN import LIDSConvNet

from ...utils import utils

__operations__ = {
    "max_pool2d": nn.MaxPool2d,
    "conv2d": SpecialConvLayer,
    "relu": nn.ReLU,
    "linear": nn.Linear,
    "batch_norm2d": nn.BatchNorm2d,
    "dropout": nn.Dropout,
    "adap_avg_pool2d": nn.AdaptiveAvgPool2d,
    "unfold": nn.Unfold,
    "fold": nn.Fold
}

class LCNCreator:
    def __init__(self, architecture, images, markers, batch_size=32, label_connected_components=True, device='cpu', superpixels_markers = None):
    
        assert(architecture is not None)
        assert(images is not None)
        assert(markers is not None)

        assert(len(images) == len(markers) and len(images) > 0)
        #structure = np.ones((3, 3, 3), dtype=np.uint8)
        #markers = label(markers, structure=structure)[0] if label_connected_components else markers
        markers = utils.label_connected_componentes(markers)

        if superpixels_markers is not None:
            indices = np.where(superpixels_markers != 0)
            markers[0, indices[0], indices[1]] = superpixels_markers[indices[0], indices[1]]
        markers = markers.astype(np.int)
        self.feature_extractor = nn.Sequential()
        #io.imsave('marker.png', utils.normalize_image(markers[0]))
        
        self.architecture = architecture
        self.images = images
        self.markers = self._prepare_markers(markers)
        self.in_channels = images[0].shape[-1]
        self.device = device 
        self.batch_size = batch_size
        
        self.counter = 0
        
        self.LCN = LIDSConvNet()

        self._build()
        
    def _build_feature_extractor(self):
        architecture = self.architecture['features']
        images = self.images
        markers = self.markers
        device = self.device
        batch_size = 32
        
        input_shape = images[0].shape
        self.last_conv_layer_out_channels = self.in_channels
        
        for key in architecture:
            layer_config = architecture[key]
            
            self._assert_params(layer_config)
            
            operation = __operations__[layer_config['operation']]
            operation_params = layer_config['params']
            
            if layer_config['operation'] == "conv2d":
                activation_config = None
                pool_config = None

                if 'activation' in layer_config:
                    activation_config = layer_config['activation']
                if 'pool' in layer_config:
                    pool_config = layer_config['pool']
                
                layer = operation(in_channels=self.last_conv_layer_out_channels, **operation_params, activation_config=activation_config, pool_config = pool_config)
                layer.initialize_weights(images, self.markers)
                    
                self.last_conv_layer_out_channels = layer.out_channels
                
            elif layer_config['operation'] == "batch_norm2d":
                layer = operation(num_features=self.last_conv_layer_out_channels)
                
            elif layer_config['operation'] == "max_pool2d":
                new_markers = []
                stride = layer_config['params']['stride']
                for marker in markers:
                    new_marker = []
                    #print(marker)
                    for old in marker:
                        x, y, label = old
                        new = (math.floor(x/stride), math.floor(y/stride), label)
                        new_marker.append(new)
                    #print(new_marker)
                    new_markers.append(new_marker)
                self.markers = new_markers
                layer = operation(**operation_params)
                
            elif layer_config['operation'] == "unfold":
                layer = operation(**operation_params)
                
            else:
                layer = operation(**operation_params)
                
            torch_images = torch.Tensor(images)
            #torch_images = (torch_images - mean)/std

            
            torch_images = torch_images.permute(0, 3, 1, 2)
            #print("Input shape:", torch_images.shape)
            input_size = torch_images.size(0)
            
            #number_of_batches = math.ceil(torch_images.size(0)/batch_size)
            
            if layer_config['operation'] != "unfold":
                outputs = torch.Tensor([])
                layer = layer.to(self.device)
                
                for i in range(0, input_size, batch_size):
                    batch = torch_images[i: i+batch_size]
                    output = layer.forward(batch.to(self.device))
                    output = output.detach().cpu()
                    outputs = torch.cat((outputs, output))
                    
                images = outputs.permute(0, 2, 3, 1).detach().numpy()
                
                #print("outpus shape*: ", outputs.size())
                
            #layer = layer.to('cpu')   
            #torch.cuda.empty_cache()
            self.LCN.feature_extractor.add_module(key, layer)
            
        self.features = images

    def _prepare_markers(self, markers):
        _markers = []
        for m in markers:
            indices = np.where(m != 0)

            max_label = m.max()
            labels = m[indices]-1 if  max_label > 1 else m[indices] 

            _markers.append([indices[0], indices[1], labels])

        return _markers
         
    def _build(self):
        self._build_feature_extractor()
            
    def _assert_params(self, params):

        if not 'operation' in params:
            raise AssertionError('Layer does not have an operation.')
        
        if not 'params' in params:
            raise AssertionError('Layer does not have operation params.')

    def get_LIDSConvNet(self):
        return self.LCN