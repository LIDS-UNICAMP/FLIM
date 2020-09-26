# noqa: D100

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from torch.nn.modules import module

from ._special_conv_layer import SpecialConvLayer
from._special_linear_layer import SpecialLinearLayer
from ._lcn import LIDSConvNet, ParallelModule

from ...utils import label_connected_components

__all__ = ['LCNCreator']

__operations__ = {
    "max_pool2d": nn.MaxPool2d,
    "conv2d": SpecialConvLayer,
    "relu": nn.ReLU,
    "linear": SpecialLinearLayer,
    "batch_norm2d": nn.BatchNorm2d,
    "dropout": nn.Dropout,
    "adap_avg_pool2d": nn.AdaptiveAvgPool2d,
    "unfold": nn.Unfold,
    "fold": nn.Fold
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
                 relabel_markers=True,
                 device='cpu',
                 superpixels_markers=None):
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
        self._input_shape = input_shape
        self._architecture = architecture
        if images is None:
            self._in_channels = input_shape[-1]
        else:
            self._in_channels = images[0].shape[-1]
            self._input_shape = list(images[0].shape)

        self._batch_size = batch_size

        self.last_conv_layer_out_channels = 0

        self.device = device
        
        self.LCN = LIDSConvNet()
        
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
        architecture = self._architecture['features']
        images = self._images
        markers = self._markers
        
        if self._relabel_markers:
            start_label = 2 if self._has_superpixel_markers else 1
            markers = label_connected_components(markers, start_label)

        if self._has_superpixel_markers:
            markers += self._superpixel_markers

        module, out_channels = self._build_module(architecture,
                                    images,
                                    markers,
                                    remove_similar_filters,
                                    similarity_level)

        self.last_conv_layer_out_channels = out_channels

        self.LCN.feature_extractor = module

    def load_model(self, state_dict):
        architecture = self._architecture

        module, out_channels = self._build_module(architecture['features'],
                                                  state_dict=state_dict)

        self.last_conv_layer_out_channels = out_channels

        self.LCN.feature_extractor = module

        if "classifier" in architecture:
            self.build_classifier(state_dict=state_dict)

        self.LCN.load_state_dict(state_dict)

    def _build_module(self,
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

        last_conv_layer_out_channels = self._input_shape[-1]
        output_shape = self._input_shape

        for key in layers_arch:
            layer_config = layers_arch[key]

            if "type" in layer_config:
                _module, last_conv_layer_out_channels = self._build_module(layer_config,
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
                        torch_images = torch.Tensor(images)

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
                        # output_shape = images.shape
            
            else:
        
                _assert_params(layer_config)
                
                operation = __operations__[layer_config['operation']]
                operation_params = layer_config['params']
                
                if layer_config['operation'] == "conv2d":
                    activation_config = None
                    pool_config = None

                    if 'activation' in layer_config:
                        activation_config = layer_config['activation']
                    if 'pool' in layer_config:
                        pool_config = layer_config['pool']

                        stride = pool_config['params']['stride']

                        output_shape[0] = output_shape[0]//stride
                        output_shape[1] = output_shape[1]//stride
                    
                    layer = operation(
                        in_channels=last_conv_layer_out_channels,
                        **operation_params,
                        activation_config=activation_config,
                        pool_config=pool_config)
                    if images is None or markers is None:
                        kernels_number = state_dict[f'feature_extractor.{key}._conv.weight'].size(0)
                        layer.initialize_weights(kernels_number=kernels_number)
                    else:
                        layer.initialize_weights(images, markers)

                    if remove_similar_filters:
                        layer.remove_similar_filters(similarity_level)
                        
                    last_conv_layer_out_channels = layer.out_channels
                    
                elif layer_config['operation'] == "batch_norm2d":
                    layer = operation(
                        num_features=last_conv_layer_out_channels)
                    
                elif layer_config['operation'] == "max_pool2d":
                    stride = operation_params['stride']

                    output_shape[0] = output_shape[0]//stride
                    output_shape[1] = output_shape[1]//stride
                    
                    layer = operation(**operation_params)
                    
                elif layer_config['operation'] == "unfold":
                    layer = operation(**operation_params)
                    
                else:
                    layer = operation(**operation_params)

                if images is not None and markers is not None:    
                    torch_images = torch.Tensor(images)

                    torch_images = torch_images.permute(0, 3, 1, 2)
                    
                    input_size = torch_images.size(0)
                    
                    if layer_config['operation'] != "unfold":
                        outputs = torch.Tensor([])
                        layer = layer.to(self.device)
                        
                        for i in range(0, input_size, batch_size):
                            batch = torch_images[i: i+batch_size]
                            output = layer.forward(batch.to(device))
                            output = output.detach().cpu()
                            outputs = torch.cat((outputs, output))
                            
                        images = outputs.permute(0, 2, 3, 1).detach().numpy()
                        # output_shape = list(images.shape)
        
                module.add_module(key, layer)
        output_shape[2] = last_conv_layer_out_channels
        self._output_shape = output_shape
       
        return module, last_conv_layer_out_channels
    
    def build_classifier(self, train_set=None, state_dict=None):
        """Buid the classifier."""

        model = self.LCN

        if model is None:
            model = LIDSConvNet()
            self.LCN = model

        classifier = model.classifier
        
        architecture = self._architecture

        features = None
        all_labels = None
        if train_set is not None:
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


        assert "classifier" in architecture, \
            "Achitecture does not specify a classifier"

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

                layer.initialize_weights(features, all_labels)
            else:
                layer = operation(**operation_params)

            if features is not None:
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
        if features is None:
            for m in classifier.modules():
                if isinstance(m, SpecialLinearLayer):
                    #nn.init.normal_(m.weight, 0, 0.01)
                    if m._linear.bias is not None:
                        nn.init.constant_(m._linear.bias, 0)   


    def update_model(self,
                     model,
                     images,
                     markers,
                     relabel_markers=True,
                     retrain=False,
                     remove_similar_filters=False,
                     similarity_level=0.85):
        """Update model with new image markers.

        Update the model feature extractor with new markers.
        It adds new filters based on new markers.
        If the old markers are used, the whole model is retrained.

        Parameters
        ----------
        model : LIDSConvNet
            A LIDSConvNet model.
        images : ndarray
            A set of images with size :math:`(N, H, W, C)`.
        markers : ndarray
            A set of image markes as label images with size :math:`(N, H, W)`.\
            The label 0 denote no label.
        relabel_markers : bool, optional
            Change markers labels so that each connected component has a \
            different label, by default True.
        retrain : bool, optional
            If False, new filters are created from the new markers.
            If True, the whole model is retrained. By default Fasle.
            Pass True if there are new images.
        remove_similar_filters : bool, optional
            Keep only one of a set of similar filters, by default False.
        similarity_level : float, optional
            A value in range :math:`(0, 1]`. \
            If filters have inner product greater than value, \
            only one of them are kept. by default 0.85.
    
        """
        assert model is not None or not isinstance(model, LIDSConvNet), \
            "A LIDSConvNet model must be provided"

        assert images is not None and markers is not None and \
            images.shape[0] > 0, "Images and markers must be provided"
        
        assert images.shape[:-1] == markers.shape, \
            "Images and markers must have compatible shapes"
        
        if retrain:
            self._images = images
            self._markers = markers
            self.build_feature_extractor(
                remove_similar_filters, similarity_level)
            return

        markers = markers.astype(np.int)

        _images = images
        old_markers = self._markers

        new_markers = markers
        mask = np.logical_and(markers != 0, old_markers != 0)
        new_markers[mask] = 0

        start_label = 2 if self._has_superpixel_markers else 1

        old_markers_relabeled = label_connected_components(old_markers, start_label)
        if self._has_superpixel_markers:
            old_markers_relabeled += self._superpixel_markers

        new_makers_relabeled = label_connected_components(new_markers, start_label)

        for _, layer in model.feature_extractor.named_children():
            if isinstance(layer, SpecialConvLayer):
                if isinstance(_images, torch.Tensor):
                    _images = _images.detach().permute(
                        0, 2, 3, 1).cpu().numpy()
                layer.update_weights(_images, old_markers_relabeled, new_makers_relabeled)

                if remove_similar_filters:
                    layer.remove_similar_filters(similarity_level)

                layer.to(self.device)
                torch_images = torch.Tensor(_images)
                torch_images = torch_images.permute(0, 3, 1, 2).to(self.device)
                _images = torch_images

            _y = layer.forward(_images)
            _images = _y

        markers[mask] = old_markers[mask]

        self._markers = markers

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
