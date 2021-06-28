# noqa: D100

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from skimage.util import view_as_windows

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from scipy.spatial import distance

import numpy as np
from torch.nn.modules import module, padding

from ._special_linear_layer import SpecialLinearLayer
from ._marker_based_norm import MarkerBasedNorm
from ._lcn import LIDSConvNet, ParallelModule
from ._decoder import Decoder

from ...utils import label_connected_components

__all__ = ['LCNCreator']

__operations__ = {
    "max_pool2d": nn.MaxPool2d,
    "avg_pool2d": nn.AvgPool2d,
    "conv2d": nn.Conv2d,
    "relu": nn.ReLU,
    "linear": SpecialLinearLayer,
    'marker_based_norm': MarkerBasedNorm,
    "batch_norm2d": nn.BatchNorm2d,
    "dropout": nn.Dropout,
    "adap_avg_pool2d": nn.AdaptiveAvgPool2d,
    "unfold": nn.Unfold,
    "fold": nn.Fold,
    "decoder": Decoder
}


class LCNCreator:

    """Class to build and a LIDSConvNet.

    LCNCreator is reponsable to build a LIDSConvNet given \
    a network architecture, a set of images, and a set of image markers.

    Attributes
    ----------
    LCN : LIDSConvNet
        The neural network built.

    last_conv_layer_out_channel : int
        The number of the last layer output channels.

    device : str
        Decive where the computaion is done.

    """

    def __init__(self,
                 architecture,
                 images=None,
                 markers=None,
                 input_shape=None,
                 batch_size=32,
                 relabel_markers=False,
                 device='cpu',
                 superpixels_markers=None,
                 remove_border=0,
                 default_std=1e-6):
        """Initialize the class.

        Parameters
        ----------
        architecture : dict
            Netwoerk's architecture specification.
        images : ndarray
            A set of images with size :math:`(N, H, W, C)`,
            by default None.
        markers : ndarray
            A set of image markes as label images with size :math:`(N, H, W)`.\
            The label 0 denote no label, by default None.
        input_shape: list
            Image shape (H, W, C), must me given if images is None. By default None.
        batch_size : int, optional
            Batch size, by default 32.
        relabel_markers : bool, optional
            Change markers labels so that each connected component has a \
            different label, by default True.
        device : str, optional
            Device where to do the computation, by default 'cpu'.
        superpixels_markers : ndarray, optional
            Extra images markers get from superpixel segmentation, \
            by default None.

        """
        assert architecture is not None

        if superpixels_markers is not None:
            self._superpixel_markers = np.expand_dims(superpixels_markers, 0).astype(np.int)
            self._has_superpixel_markers = True

        else:
           self._has_superpixel_markers = False 

        if markers is not None:
            markers = markers.astype(np.int)

        self._feature_extractor = nn.Sequential()
        self._relabel_markers = relabel_markers
        self._images = images
        self._markers = markers
        self._input_shape = np.array(input_shape)
        self._architecture = architecture
    
        if images is None:
            self._in_channels = input_shape[-1]
        else:
            self._in_channels = images[0].shape[-1]
            self._input_shape = list(images[0].shape)

        self._batch_size = batch_size

        self.last_conv_layer_out_channels = 0

        self.device = device

        self._remove_border = remove_border

        self._default_std = default_std
        

        self._outputs = dict()

        self._skips = _find_skip_connections(self._architecture)
        self._to_save_outputs = _find_outputs_to_save(self._skips)
        
        self.LCN = LIDSConvNet(skips=self._skips, outputs_to_save=self._to_save_outputs, remove_boder=remove_border)

    def build_feature_extractor(self,
                                remove_similar_filters=False,
                                similarity_level=0.85):
        """Buid the feature extractor.

        If there is a special convolutional layer, \
        it will be initialize with weights learned from image markers.

        Parameters
        ----------
        remove_similar_filters : bool, optional
            Keep only one of a set of similar filters, by default False.
        similarity_level : float, optional
            A value in range :math:`(0, 1]`. \
            If filters have inner product greater than value, \
            only one of them are kept. by default 0.85.

        """
        self._output_shape = self._input_shape

        if "features" in self._architecture:
            architecture = self._architecture['features']
            images = self._images
            markers = self._markers

            if "input" in self._to_save_outputs:
                self._outputs['input'] = images
            
            if self._relabel_markers and markers is not None:
                start_label = 2 if self._has_superpixel_markers else 1
                markers = label_connected_components(markers, start_label)

            if self._has_superpixel_markers:
                markers += self._superpixel_markers

            module, out_channels, _, _ = self._build_module(None,
                                        architecture,
                                        images,
                                        markers,
                                        remove_similar_filters=remove_similar_filters,
                                        similarity_level=similarity_level)

            self.last_conv_layer_out_channels = out_channels

            self.LCN.feature_extractor = module

        torch.cuda.empty_cache()

    def load_model(self, state_dict):
        architecture = self._architecture

        module, out_channels = self._build_module("", architecture['features'],
                                                  state_dict=state_dict)

        self.last_conv_layer_out_channels = out_channels

        self.LCN.feature_extractor = module

        if "classifier" in architecture:
            self.build_classifier(state_dict=state_dict)

        self.LCN.load_state_dict(state_dict)

    def _build_module(self,
                      module_name,
                      module_arch,
                      images=None,
                      markers=None,
                      state_dict=None,
                      remove_similar_filters=False,
                      similarity_level=0.85):
        """Builds a module.

        A module can have submodules.

        Parameters
        ----------
        module_arch : dict
            module configuration
        images : ndarray
            A set of images with size :math:`(N, H, W, C)`.
        markers : ndarray
            A set of image markes as label images with size :math:`(N, H, W)`.\
            The label 0 denote no label.
        state_dict: OrderedDict
            If images and markers are None, this argument must be given,\
            by default None.
        ----------
        remove_similar_filters : bool, optional
            Keep only one of a set of similar filters, by default False.
        similarity_level : float, optional
            A value in range :math:`(0, 1]`. \
            If filters have inner product greater than value, \
            only one of them are kept. by default 0.85.

        Returns
        -------
        nn.Module
            A PyTorch module.
        """        
        device = self.device

        batch_size = self._batch_size

        module_type = module_arch['type']

        if module_type == 'parallel':
            module = ParallelModule()
        else:
            module = nn.Sequential()
        
        layers_arch = module_arch['layers']

        output_shape = self._input_shape if images is None else np.array(images[0].shape)
        last_conv_layer_out_channels = output_shape[-1] 

        for key in layers_arch:
            new_module_name = key if module_name is None else f"{module_name}.{key}"

            if new_module_name in self._skips:
                inputs_names = self._skips[new_module_name]
                images = np.concatenate([images, *[self._outputs[name] for name in inputs_names]], axis=-1)
                output_shape = np.array(images[0].shape)
                last_conv_layer_out_channels = output_shape[-1]

            layer_config = layers_arch[key]
            print(f"Building {key}")
        
            if "type" in layer_config:
                _module, last_conv_layer_out_channels, images, markers = self._build_module(new_module_name,
                                                                           layer_config,
                                                                           images,
                                                                           markers,
                                                                           state_dict,
                                                                           remove_similar_filters,
                                                                           similarity_level)
                if module_type == 'parallel':
                    module.append(_module)
                else:
                    module.add_module(key, _module)

                    if images is not None and markers is not None:
                        '''torch_images = torch.Tensor(images)

                        torch_images = torch_images.permute(0, 3, 1, 2)
                        
                        input_size = torch_images.size(0)
                        
                        outputs = torch.Tensor([])
                        _module = _module .to(self.device)
                        
                        for i in range(0, input_size, batch_size):
                            batch = torch_images[i: i+batch_size]
                            output = _module.forward(batch.to(device))
                            output = output.detach().cpu()
                            outputs = torch.cat((outputs, output))

                        # last_conv_layer_out_channels = outputs.size(1)
                        images = outputs.permute(0, 2, 3, 1).detach().numpy()
                        # output_shape = images.shape'''
                        
                        if new_module_name in self._to_save_outputs:
                            self._outputs[new_module_name] = images
                        
            else:
        
                _assert_params(layer_config)
                
                operation = __operations__[layer_config['operation']]
                operation_params = layer_config['params']
                
                if layer_config['operation'] == "conv2d":

                    number_of_kernels_per_marker = operation_params.get("number_of_kernels_per_marker", 16)
                    use_random_kernels = operation_params.get("use_random_kernels", False)
                    use_pca = operation_params.get("use_pca", False)

    
                    kernel_size = operation_params['kernel_size']
                    stride = operation_params.get('stride', 0)
                    padding = operation_params.get('padding', 0)
                    padding_mode = operation_params.get('padding_mode', 'zeros')
                    dilation = operation_params.get('dilation', 0)
                    groups = operation_params.get('groups', 1)
                    bias = operation_params.get('bias', False)
                
                    out_channels = operation_params.get('out_channels', None)
                    
                    in_channels = last_conv_layer_out_channels

                    if isinstance(dilation, int):
                        dilation = [dilation, dilation]

                    if isinstance(padding, int):
                        padding = [padding, padding]

                    if isinstance(kernel_size, int):
                        kernel_size = [kernel_size, kernel_size]

                    if markers is not None and "number_of_kernels_per_marker" not in operation_params:
                        number_of_kernels_per_marker = math.ceil(operation_params["out_channels"]/np.array(markers).max())

                    default_std=self._default_std

                    if (images is None or markers is None) and state_dict is not None:
                        out_channels = state_dict[f'feature_extractor.{key}.weight'].size(0)

                    weights = _initialize_conv2d_weights(images,
                                                         markers,
                                                         in_channels,
                                                         out_channels=out_channels,
                                                         kernel_size=kernel_size,
                                                         dilation=dilation,
                                                         number_of_kernels_per_marker=number_of_kernels_per_marker,
                                                         use_random_kernels=use_random_kernels,
                                                         default_std=default_std)
                                                         
                    if out_channels is None:
                        out_channels = weights.shape[0]
                    
                    layer = nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size,
                                      stride,
                                      padding,
                                      dilation,
                                      groups,
                                      bias,
                                      padding_mode)

                    layer.weight = nn.Parameter(torch.from_numpy(weights))

                    if remove_similar_filters:
                        layer = _remove_similar_filters(layer, similarity_level)
                        
                    last_conv_layer_out_channels = layer.out_channels

                elif layer_config['operation'] == 'marker_based_norm':

                    if images is None or markers is None:
                        mean = None
                        std = None
                    else:
                        kernel_size = operation_params['kernel_size']
                        dilation = operation_params.get('dilation', 0)
                        in_channels = last_conv_layer_out_channels

                        if isinstance(dilation, int):
                            dilation = [dilation, dilation]

                        if isinstance(kernel_size, int):
                            kernel_size = [kernel_size, kernel_size]

                        patches, _ = _generate_patches(images,
                                                       markers,
                                                       in_channels,
                                                       kernel_size,
                                                       dilation)

        
                        mean = torch.from_numpy(patches.mean(axis=(0, 1, 2), keepdims=True)).view(1, -1, 1, 1).float()
                        std = torch.from_numpy(patches.std(axis=(0, 1, 2), keepdims=True)).view(1, -1, 1, 1).float()
                    
                    layer = MarkerBasedNorm(mean=mean,
                                            std=std,
                                            in_channels=last_conv_layer_out_channels,
                                            default_std=self._default_std)

                elif layer_config['operation'] == "batch_norm2d":
                    layer = operation(
                        num_features=last_conv_layer_out_channels)
                    layer.train()
                    layer = layer.to(device)
                    if images is not None and markers is not None:    
                        torch_images = torch.Tensor(images)

                        torch_images = torch_images.permute(0, 3, 1, 2)
                        
                        input_size = torch_images.size(0)
                        
                        for i in range(0, input_size, batch_size):
                            batch = torch_images[i: i+batch_size]
                            output = layer.forward(batch.to(device))
                        
                    layer.eval()
                    
                elif layer_config['operation'] == "max_pool2d" or layer_config['operation'] == "avg_pool2d":
                    stride = operation_params['stride']
                    kernel_size = operation_params['kernel_size']
                    
                    if 'padding' in operation_params:
                        padding = operation_params['padding']
                        if isinstance(padding, int):
                            padding = [padding, padding]
                    else:
                        padding = [0, 0]
                    
                    if isinstance(kernel_size, int):
                        kernel_size = [kernel_size, kernel_size]

                    output_shape[0] = math.floor((output_shape[0] + 2*padding[0] - kernel_size[0])/stride + 1)
                    output_shape[1] = math.floor((output_shape[1] + 2*padding[1] - kernel_size[1])/stride + 1)
                    
                    operation_params['stride'] = 1
                    _layer = operation(**operation_params)

                    if images is not None and markers is not None:    
                        torch_images = torch.Tensor(images)

                        torch_images = torch_images.permute(0, 3, 1, 2)
                        
                        input_size = torch_images.size(0)
                        
                        outputs = torch.Tensor([])
                        layer = _layer.to(self.device)
                        
                        for i in range(0, input_size, batch_size):
                            batch = torch_images[i: i+batch_size]
                            output = _layer.forward(batch.to(device))
                            output = output.detach().cpu()
                            outputs = torch.cat((outputs, output))
                            
                        images = outputs.permute(0, 2, 3, 1).detach().numpy()
                    
                    #if markers is not None:
                    #    markers = _pooling_markers(markers, kernel_size, stride=stride, padding=padding)

                    operation_params['stride'] = stride
                    layer = operation(**operation_params)
                    
                elif layer_config['operation'] == "adap_avg_pool2d":
                    output_shape[0] = operation_params['output_size'][0]
                    output_shape[1] = operation_params['output_size'][1]
                    
                    layer = operation(**operation_params)
                    
                elif layer_config['operation'] == "unfold":
                    layer = operation(**operation_params)

                    torch_image = torch.from_numpy(images[0])
                    torch_image = torch_image.unsqueeze(0)
                    torch_image = torch_image.permute(0, 3, 1, 2).to(device)
                    layer.train()
                    layer.to(device)

                    output = layer(torch_image)

                    output_shape[0] = 1
                    output_shape[1] = 1
                    output_shape[2] = output.shape[1]

                    last_conv_layer_out_channels = output.shape[1]

                elif layer_config['operation'] == 'decoder':
                    layer = operation(images, self._markers, device=device, **operation_params)
                    layer.to(device)

                else:
                    layer = operation(**operation_params)
                    
                if images is not None and markers is not None:    
                    torch_images = torch.Tensor(images)

                    torch_images = torch_images.permute(0, 3, 1, 2)
                    
                    input_size = torch_images.size(0)
                    
                    if layer_config['operation'] != "unfold" and not ('pool' in layer_config['operation']):
                        outputs = torch.Tensor([])
                        layer = layer.to(self.device)
                        
                        for i in range(0, input_size, batch_size):
                            batch = torch_images[i: i+batch_size]
                            output = layer.forward(batch.to(device))
                            output = output.detach().cpu()
                            outputs = torch.cat((outputs, output))
                            
                        images = outputs.permute(0, 2, 3, 1).detach().numpy()
                        # output_shape = list(images.shape)

                        if new_module_name in self._to_save_outputs:
                            self._outputs[new_module_name] = images
                layer.train()
                module.add_module(key, layer)


        output_shape[2] = last_conv_layer_out_channels

        if self._remove_border > 0:
            output_shape[0] -= 2*self._remove_border
            output_shape[1] -= 2*self._remove_border

        self._output_shape = output_shape
        
        return module, last_conv_layer_out_channels, images, markers
    
    def build_classifier(self, train_set=None, state_dict=None):
        """Buid the classifier."""

        model = self.LCN

        if model is None:
            model = LIDSConvNet(self._remove_border, self._skips, self._to_save_outputs)
            self.LCN = model

        classifier = model.classifier
        
        architecture = self._architecture


        assert "classifier" in architecture, \
            "Achitecture does not specify a classifier"
            
        features = None
        all_labels = None
        use_backpropagation = 'backpropagation' in architecture['classifier'] and architecture['classifier']['backpropagation']
        
        if train_set is not None and not use_backpropagation:
            loader = DataLoader(train_set, self._batch_size, shuffle=False)
            for inputs, labels in loader:
                inputs = inputs.to(self.device)

                outputs = model.feature_extractor(inputs).detach().cpu().flatten(1)

                if features is None:
                    features = outputs
                    all_labels = labels
                else:
                    features = torch.cat((features, outputs))
                    all_labels = torch.cat((all_labels, labels))
        
            features = features.numpy()
            all_labels.numpy()

        cls_architecture = architecture['classifier']['layers']

        for key in cls_architecture:
            layer_config = cls_architecture[key]
            
            operation = __operations__[layer_config['operation']]
            operation_params = layer_config['params']
            
            if layer_config['operation'] == 'linear':
                if operation_params['in_features'] == -1:
                    operation_params['in_features'] = np.prod(self._output_shape)
                if train_set is None and state_dict is not None:
                    weights = state_dict[f'classifier.{key}._linear.weight']
                    operation_params['in_features'] = weights.shape[1]
                    operation_params['out_features'] = weights.shape[0]

                layer = operation(**operation_params)

                if use_backpropagation:
                    layer.initialize_weights()
                else:
                    layer.initialize_weights(features, all_labels)
            else:
                layer = operation(**operation_params)

            if features is not None and not use_backpropagation:
                torch_features = torch.Tensor(features)
                
                input_size = torch_features.size(0)
                
                outputs = torch.Tensor([])
                _module = layer.to(self.device)
                
                for i in range(0, input_size, self._batch_size):
                    batch = torch_features[i: i+self._batch_size]
                    output = _module.forward(batch.to(self.device))
                    output = output.detach().cpu()
                    outputs = torch.cat((outputs, output))
                
                features = outputs.numpy()

            classifier.add_module(key, layer)

        #initialization
        if features is None or use_backpropagation:
            for m in classifier.modules():
                if isinstance(m, SpecialLinearLayer):
                    m._linear.weight.data.normal_(0, 0.01)
                    if m._linear.bias is not None:
                        nn.init.constant_(m._linear.bias, 0)   
        torch.cuda.empty_cache()

    def remove_filters(self, layer_index, filter_indices):
        """Remove layer's filters.

        Be aware that following layers must be updated.

        Parameters
        ----------
        layer_index : int
            Layer index.
        filter_indices : ndarray
            A 1D array with indices to remove.
        """

        layer = self.LCN.feature_extractor[layer_index]
        
        assert isinstance(layer, SpecialConvLayer),\
            "Layer is not a Special Conv Layer"

        layer.remove_filters(filter_indices)

    def get_LIDSConvNet(self):
        """Get the LIDSConvNet built.

        Returns
        -------
        LIDSConvNet
            The neural network built.

        """
        return self.LCN


def _prepare_markers(markers):
    """Convert image markers to the expected format.

    Convert image markers from label images to a list of coordinates.


    Parameters
    ----------
    markers : ndarray
        A set of image markers as image labels with size :math:`(N, H, W)`.

    Returns
    -------
    list[ndarray]
        Image marker as a list of coordinates.
        For each image there is an ndarry with shape \
        :math:`3 \times N` where :math:`N` is the number of markers pixels.
        The first row is the markers pixels :math:`x`-coordinates, \
        second row is the markers pixels :math:`y`-coordinates,
        and the third row is the markers pixel labels.

    """
    _markers = []
    for m in markers:
        indices = np.where(m != 0)

        max_label = m.max()
        labels = m[indices]-1 if max_label > 1 else m[indices]

        _markers.append([indices[0], indices[1], labels])

    return _markers


def _assert_params(params):
    """Check network's architecture specification.

    Check if the network's architecture specification has \
    the fields necessary to build the network.

    Parameters
    ----------
    params : dict
        The parameters for building a layer.

    Raises
    ------
    AssertionError
        If a operation is not specified.
    AssertionError
        If operation parameters are not specified.
        
    """
    if 'operation' not in params:
        raise AssertionError('Layer does not have an operation.')
    
    if 'params' not in params:
        raise AssertionError('Layer does not have operation params.')

def _pooling_markers(markers, kernel_size, stride=1, padding=0):
    new_markers = []
    for marker in markers:
      indices_x, indices_y = np.where(marker != 0)
      
      marker_shape = [*marker.shape]

      marker_shape[0] = math.floor((marker_shape[0] + 2*padding[0] - kernel_size[0])/stride + 1)
      marker_shape[1] = math.floor((marker_shape[1] + 2*padding[1] - kernel_size[1])/stride + 1)

      new_marker = np.zeros(marker_shape, dtype=np.int)
      x_limit = marker.shape[0] + 2*padding[0] - kernel_size[0]
      y_limit = marker.shape[1] + 2*padding[1] - kernel_size[1]
      for x, y in zip(indices_x, indices_y):
          if x > x_limit or y > y_limit:
            continue
          new_marker[x//stride][y//stride] = marker[x][y]

      new_markers.append(new_marker)

    return np.array(new_markers)


def _find_skip_connections_in_module(module_name, module):
    skips = dict()

    layers = module['layers']

    for layer_name, layer_config in layers.items():
        key_name = f"{module_name}.{layer_name}" if module_name is not None else layer_name

        if 'type' in layer_config:
            submodules_skips = _find_skip_connections_in_module(key_name, layer_config)

            skips.update(submodules_skips)
            
        if "inputs" in layer_config:
            skips[key_name] = layer_config['inputs']
    
    return skips

def _find_skip_connections(arch):
    if "features" in arch:
        arch = arch['features']

    skips = _find_skip_connections_in_module(None, arch)

    return skips

def _find_outputs_to_save(skips):
    outputs_to_save = {}
    for _, inputs in skips.items():
        
        for layer_name in inputs:
            outputs_to_save[layer_name] = True

    return outputs_to_save


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

def _generate_patches(images,
                      markers,
                      in_channels,
                      kernel_size,
                      dilation):
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
            dilation = np.array(dilation)

            dilated_kernel_size = kernel_size + (dilation - 1) * (kernel_size-1)
            dilated_padding = dilated_kernel_size // 2
            image_pad = np.pad(image, ((dilated_padding[0], dilated_padding[0]),
                                    (dilated_padding[1], dilated_padding[1]), (0, 0)),
                            mode='constant', constant_values=0)

            patches = view_as_windows(image_pad,
                                      (dilated_kernel_size[0], dilated_kernel_size[1], in_channels),
                                      step=1)
                                      
            if dilation[0] > 1 or dilation[1] > 0:
                r = np.arange(0, dilated_kernel_size[0], dilation[0])
                s = np.arange(0, dilated_kernel_size[1], dilation[1])
                patches = patches[:, :, :, r, : , :][:, :, :, :, s , :]

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


def _calculate_conv2d_weights(images, markers, in_channels, kernel_size, dilation, number_of_kernels_per_marker, default_std):
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
        patches, labels = _generate_patches(images,
                                            markers,
                                            in_channels,
                                            kernel_size,
                                            dilation)

        
        mean_by_channel = patches.mean(axis=(0, 1, 2), keepdims=True)
        std_by_channel = patches.std(axis=(0, 1, 2), keepdims=True)
            
        patches = (patches - mean_by_channel)/(std_by_channel + default_std)

        kernel_weights = _kmeans_roots(patches,
                                       labels,
                                       number_of_kernels_per_marker)

        kernels_shape = kernel_weights.shape
        kernel_weights = kernel_weights.reshape(kernels_shape[0], -1)
        norm = np.linalg.norm(kernel_weights, axis=1)
        norm = np.expand_dims(norm, 1)
        #print(norm)
        kernel_weights = kernel_weights/(norm + 0.00001)
        kernel_weights = kernel_weights.reshape(kernels_shape)
        return kernel_weights


def _initialize_conv2d_weights(images=None,
                               markers=None,
                               in_channels=3,
                               out_channels=None,
                               kernel_size=None,
                               dilation=[1, 1],
                               number_of_kernels_per_marker=16,
                               use_random_kernels=False,
                               default_std=0.1):
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
        if use_random_kernels:
            kernels_weights = _create_random_pca_kernels(n=out_channels * 10,
                                                         k=out_channels,
                                                         in_channels=in_channels,
                                                         kernel_size=kernel_size)
            
        elif images is not None and markers is not None:
            kernels_weights = _calculate_conv2d_weights(images,
                                                        markers,
                                                        in_channels,
                                                        kernel_size,
                                                        dilation,
                                                        number_of_kernels_per_marker,
                                                        default_std=default_std)
            kernels_weights = np.rollaxis(kernels_weights, 3, 1)

            if  out_channels is not None and out_channels < kernels_weights.shape[0] and np.prod(kernels_weights.shape[1:]) > out_channels:
                kernels_weights = _select_kernels_with_pca(kernels_weights, out_channels)
        
            elif out_channels is not None and out_channels < kernels_weights.shape[0]:
                kernels_weights = _kmeans_roots(kernels_weights, np.ones(kernels_weights.shape[0]), out_channels)

        else:
            kernels_weights = torch.rand(out_channels,
                                         in_channels,
                                         kernel_size[0],
                                         kernel_size[1]).numpy()

        return kernels_weights

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

def _remove_similar_filters(layer, similarity_level=0.85):
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

        filters = layer.weight

        similarity_matrix = _compute_similarity_matrix(filters)

        keep_filter = np.full(filters.size(0), True, np.bool)

        for i in range(0, filters.size(0)):
            if keep_filter[i]:

                mask = similarity_matrix[i] >= similarity_level
                indices = np.where(mask)

                keep_filter[indices] = False
        
        selected_filters = filters[keep_filter]

        out_channels = selected_filters.size(0)

        new_conv = nn.Conv2d(layer.in_channels,
                             out_channels,
                             kernel_size=layer.kernel_size,
                             stride=layer.stride,
                             bias=layer.bias,
                             padding=layer.padding)

        new_conv.weight = nn.Parameter(selected_filters)

        return new_conv

