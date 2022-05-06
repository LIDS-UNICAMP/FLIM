# noqa: D100

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance
from skimage.util import view_as_windows
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import (
    _euclidean_distances,
    cosine_similarity,
    euclidean_distances,
)
from torch.nn.modules.loss import CrossEntropyLoss

from ..._constants import DIVISION_EPSILON
from ...utils import label_connected_components
from ._decoder import Decoder
from ._lcn import LIDSConvNet, ParallelModule
from ._marker_based_norm import MarkerBasedNorm2d, MarkerBasedNorm3d

__all__ = ["LCNCreator"]

# TODO I think adaptative pooling 2D is missing

__operations__ = {
    "max_pool2d": nn.MaxPool2d,
    "max_pool3d": nn.MaxPool3d,
    "avg_pool2d": nn.AvgPool2d,
    "avg_pool3d": nn.AvgPool3d,
    "conv2d": nn.Conv2d,
    "conv3d": nn.Conv3d,
    "relu": nn.ReLU,
    "linear": nn.Linear,
    "marker_based_norm": MarkerBasedNorm2d,
    "m_norm2d": MarkerBasedNorm2d,
    "m_norm3d": MarkerBasedNorm3d,
    "batch_norm2d": nn.BatchNorm2d,
    "batch_norm3d": nn.BatchNorm3d,
    "dropout": nn.Dropout,
    "adap_avg_pool3d": nn.AdaptiveAvgPool3d,
    "unfold": nn.Unfold,
    "fold": nn.Fold,
    "decoder": Decoder,
}


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight=None):
        """if smoothing == 0, it's one-hot method
        if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


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

    def __init__(
        self,
        architecture,
        images=None,
        markers=None,
        input_shape=None,
        batch_size=32,
        relabel_markers=False,
        device="cpu",
        superpixels_markers=None,
        remove_border=0,
        random_state=None,
        multilevel_clustering=True,
    ):
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
            self._superpixel_markers = np.expand_dims(superpixels_markers, 0).astype(
                np.int
            )
            self._has_superpixel_markers = True

        else:
            self._has_superpixel_markers = False

        if markers is not None:
            markers = markers.astype(int)

        self._feature_extractor = nn.Sequential()
        self._relabel_markers = relabel_markers
        self._images = images
        self._markers = markers
        self._input_shape = input_shape
        self._architecture = architecture
        self._multilevel_clustering = multilevel_clustering

        if images is None:
            self._in_channels = input_shape[-1]
        else:
            self._in_channels = images[0].shape[-1]
            self._input_shape = list(images[0].shape)

        self._batch_size = batch_size

        self.last_conv_layer_out_channels = 0

        self.device = device

        self._remove_border = remove_border

        self.random_state = random_state

        self._outputs = dict()

        self._skips = _find_skip_connections(self._architecture)
        self._to_save_outputs = _find_outputs_to_save(self._skips)

        self.LCN = LIDSConvNet(
            skips=self._skips,
            outputs_to_save=self._to_save_outputs,
            remove_boder=remove_border,
        )

    def build_model(
        self,
        remove_similar_filters: bool = False,
        similarity_level: float = 0.85,
        verbose: bool = False,
    ):
        """Build the model.

        Parameters 
        ----------
        remove_similar_filters : bool, optional
            Keep only one of a set of similar filters, by default False.
        similarity_level : float, optional
            A value in range :math:`(0, 1]`. \
            If filters have inner product greater than value, \
            only one of them are kept. by default 0.85.

        """

        architecture = self._architecture
        images = self._images
        markers = self._markers

        if self._relabel_markers and markers is not None:
            start_label = 2 if self._has_superpixel_markers else 1
            markers = label_connected_components(
                markers, start_label, is_3d=markers.ndim == 4
            )

        if self._has_superpixel_markers:
            markers += self._superpixel_markers
        input_shape = self._input_shape
        # if len(input_shape) == 1:
        #    input_shape = [0, 0, *input_shape]
        if images is not None:
            input_shape = images[0].shape

        for module_name, module_arch in architecture.items():

            if "input" in self._to_save_outputs:
                self._outputs["input"] = images

            module, module_output_shape, images, markers = self._build_module(
                module_name,
                module_arch,
                images,
                markers,
                input_shape,
                remove_similar_filters=remove_similar_filters,
                similarity_level=similarity_level,
            )

            input_shape = module_output_shape
            # self.last_conv_layer_out_channels = out_channels

            self.LCN.add_module(module_name, module)

        # TODO is it necessary to empty cuda memory?
        torch.cuda.empty_cache()

    def build_feature_extractor(
        self, remove_similar_filters=False, similarity_level=0.85
    ):
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
        warnings.warn(
            "This function is deprecated. "
            "Please use the 'build_model' method instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self._output_shape = self._input_shape

        architecture = self._architecture
        images = self._images
        markers = self._markers

        if "input" in self._to_save_outputs:
            self._outputs["input"] = images

        if self._relabel_markers and markers is not None:
            start_label = 2 if self._has_superpixel_markers else 1
            markers = label_connected_components(
                markers, start_label, is_3d=markers.ndim == 4
            )

        if self._has_superpixel_markers:
            markers += self._superpixel_markers

        module, out_channels, _, _ = self._build_module(
            None,
            architecture,
            images,
            markers,
            remove_similar_filters=remove_similar_filters,
            similarity_level=similarity_level,
        )

        self.last_conv_layer_out_channels = out_channels

        self.LCN.feature_extractor = module

        torch.cuda.empty_cache()

    def load_model(self, state_dict):
        architecture = self._architecture

        module, out_channels = self._build_module(
            "", architecture["features"], state_dict=state_dict
        )

        self.last_conv_layer_out_channels = out_channels

        self.LCN.feature_extractor = module

        if "classifier" in architecture:
            self.build_classifier(state_dict=state_dict)

        self.LCN.load_state_dict(state_dict)

    def _build_module(
        self,
        module_name,
        module_arch,
        images=None,
        markers=None,
        input_shape=None,
        remove_similar_filters=False,
        similarity_level=0.85,
        verbose=True,
    ):
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

        # assume that module is sequential
        module_type = module_arch.get("type", "sequential")

        module_params = module_arch.get("params", {})

        if module_type == "parallel":
            aggregate_fn = module_params["aggregate"]
            module = ParallelModule(aggregate_fn=aggregate_fn)
        else:
            module = nn.Sequential()

        layers_arch = module_arch["layers"]

        module_output_shape = None

        for key in layers_arch:
            new_module_name = key if module_name is None else f"{module_name}.{key}"

            layer_config = layers_arch[key]
            if verbose:
                print(f"Building {key}")

            if "type" in layer_config:
                _module, _layer_output_shape, images, markers = self._build_module(
                    new_module_name,
                    layer_config,
                    images,
                    markers,
                    input_shape,
                    remove_similar_filters,
                    similarity_level,
                )

                module.add_module(key, _module)
                input_shape = _layer_output_shape

            else:

                _assert_params(layer_config)

                operation = __operations__[layer_config["operation"]]
                operation_params = layer_config["params"]

                if (
                    layer_config["operation"] == "conv2d"
                    or layer_config["operation"] == "conv3d"
                ):
                    # if bias then set training params
                    if operation_params.get("bias", False) is True:
                        if "epochs" not in operation_params:
                            operation_params["epochs"] = module_params.get("epochs", 50)
                        if "lr" not in operation_params:
                            operation_params["lr"] = module_params.get("lr", 0.001)
                        if "wd" not in operation_params:
                            operation_params["wd"] = module_params.get("wd", 0.9)

                    layer = self._build_conv_layer(
                        images,
                        markers,
                        remove_similar_filters,
                        similarity_level,
                        input_shape,
                        layer_config,
                    )
                    is_3d = layer_config["operation"] == "conv3d"
                    end = 3 if is_3d else 2

                    if len(input_shape) > 1:
                        _layer_output_shape = [*input_shape[:end], layer.out_channels]

                        spatial_size = np.array(input_shape[:end])
                        kernel_size = np.array(layer.kernel_size)
                        stride = np.array(layer.stride)
                        padding = np.array(layer.padding)
                        dilation = np.array(layer.dilation)
                        _layer_output_shape[:end] = list(
                            np.floor(
                                (
                                    spatial_size
                                    + 2 * padding
                                    - dilation * (kernel_size - 1)
                                    - 1
                                )
                                / stride
                                + 1
                            ).astype(int)
                        )
                    else:
                        _layer_output_shape = [layer.out_channels]

                elif layer_config["operation"] in [
                    "marker_based_norm",
                    "m_norm2d",
                    "m_norm3d",
                ]:
                    if layer_config["operation"] == "marker_based_norm":
                        warnings.warn(
                            "'marker_based_norm' operation name has been renamed to 'm_norm2d'",
                            DeprecationWarning,
                            stacklevel=2,
                        )

                    layer = self._build_m_norm_layer(
                        images, markers, input_shape, layer_config
                    )

                    _layer_output_shape = input_shape

                elif (
                    layer_config["operation"] == "batch_norm2d"
                    or layer_config["operation"] == "batch_norm3d"
                ):
                    operation_params = layer_config["params"]
                    operation_params["num_features"] = input_shape[-1]

                    layer = operation(**operation_params)
                    layer.train()
                    layer = layer.to(device)
                    is_3d = "3d" in layer_config["operation"]
                    if images is not None and markers is not None:
                        torch_images = torch.Tensor(images)

                        if is_3d:
                            torch_images = torch_images.permute(0, 4, 3, 1, 2)
                        else:
                            torch_images = torch_images.permute(0, 3, 1, 2)

                        input_size = torch_images.size(0)

                        for i in range(0, input_size, batch_size):
                            batch = torch_images[i : i + batch_size]
                            output = layer.forward(batch.to(device))

                    layer.eval()

                elif (
                    layer_config["operation"] == "max_pool2d"
                    or layer_config["operation"] == "avg_pool2d"
                    or layer_config["operation"] == "max_pool3d"
                    or layer_config["operation"] == "avg_pool3d"
                ):

                    layer, _ = self._build_pool_layer(
                        images, markers, batch_size, layer_config
                    )

                    is_3d = "3d" in layer_config["operation"]
                    end = 3 if is_3d else 2

                    _layer_output_shape = [*input_shape]

                    if len(input_shape) == (end + 1):
                        spatial_size = np.array(input_shape[:end])
                        kernel_size = np.array(layer.kernel_size)
                        stride = np.array(layer.stride)
                        padding = np.array(layer.padding)
                        if "avg" in layer_config["operation"]:
                            dilation = np.array([1] * end)
                        else:
                            dilation = np.array(layer.dilation)
                        _layer_output_shape[:end] = list(
                            np.floor(
                                (
                                    spatial_size
                                    + 2 * padding
                                    - dilation * (kernel_size - 1)
                                    - 1
                                )
                                / stride
                                + 1
                            ).astype(int)
                        )

                elif (
                    layer_config["operation"] == "adap_avg_pool2d"
                    or layer_config["operation"] == "adap_avg_pool3d"
                ):
                    output_size = operation_params["output_size"]
                    if len(output_size) == 2:
                        _layer_output_shape = np.array(output_size, input_shape[-1])

                    layer = operation(**operation_params)

                elif layer_config["operation"] == "unfold":
                    # TODO support 3D images?
                    kernel_size = operation_params["kernel_size"]
                    stride = operation_params.get("stride", 1)
                    padding = operation_params.get("padding", 0)
                    dilation = operation_params.get("dilation", 1)

                    is_3d = len(input_shape) == 4

                    if isinstance(kernel_size, int):
                        kernel_size = np.array([kernel_size] * (3 if is_3d else 2))
                    if isinstance(stride, int):
                        stride = np.array([stride] * (3 if is_3d else 2))
                    if isinstance(padding, int):
                        padding = np.array([padding] * (3 if is_3d else 2))
                    if isinstance(dilation, int):
                        dilation = np.array([dilation] * (3 if is_3d else 2))

                    layer = operation(**operation_params)
                    _layer_output_shape = [*input_shape]
                    spatial_size = np.array(input_shape[:-1])

                    output_spatial_size = np.prod(
                        np.floor(
                            (
                                spatial_size
                                + 2 * padding
                                - dilation * (kernel_size - 1)
                                - 1
                            )
                            / stride
                            + 1
                        )
                    )

                    _layer_output_shape[0] = output_spatial_size[0]
                    _layer_output_shape[1] = output_spatial_size[1]

                    if is_3d:
                        _layer_output_shape[2] = output_spatial_size[2]

                elif layer_config["operation"] == "decoder":
                    if images.ndim > 4:
                        raise NotImplementedError(
                            "FLIM decoder does not currently support 3D images."
                        )
                    layer = operation(
                        images, self._markers, device=device, **operation_params
                    )
                    layer.to(device)

                elif layer_config["operation"] == "linear":
                    if operation_params["in_features"] == -1:
                        operation_params["in_features"] = np.prod(input_shape)
                        _flatten_layer = nn.Flatten()
                        module.add_module("flatten", _flatten_layer)

                    layer = operation(**operation_params)
                    # initialization
                    nn.init.xavier_normal_(layer.weight, nn.init.calculate_gain("relu"))
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

                    layer.to(device)
                    _layer_output_shape = [operation_params["out_features"]]

                else:
                    layer = operation(**operation_params)
                    _layer_output_shape = [*input_shape]

                if images is not None and markers is not None:
                    torch_images = torch.Tensor(images)
                    is_3d = torch_images.ndim == 5
                    if is_3d:
                        torch_images = torch_images.permute(0, 4, 3, 1, 2)
                    else:
                        torch_images = torch_images.permute(0, 3, 1, 2)

                    input_size = torch_images.size(0)

                    if (
                        layer_config["operation"] != "unfold"
                        and not ("pool" in layer_config["operation"])
                        and not ("linear" in layer_config["operation"])
                    ):
                        outputs = torch.Tensor([])
                        layer = layer.to(self.device)

                        for i in range(0, input_size, batch_size):
                            batch = torch_images[i : i + batch_size]
                            output = layer.forward(batch.to(device))
                            output = output.detach().cpu()
                            outputs = torch.cat((outputs, output))
                        if is_3d:
                            images = outputs.permute(0, 3, 4, 2, 1).detach().numpy()
                        else:
                            images = outputs.permute(0, 2, 3, 1).detach().numpy()

                        # _layer_output_shape = list(images.shape[1:])
                layer.train()
                module.add_module(key, layer)

                if module_type == "sequential":
                    module_output_shape = _layer_output_shape
                    input_shape = _layer_output_shape
                else:
                    if aggregate_fn == "concat":
                        if module_output_shape is None:
                            module_output_shape = [*_layer_output_shape]
                        else:
                            module_output_shape[-1] += _layer_output_shape[-1]
                    else:
                        module_output_shape = _layer_output_shape

        if self._remove_border > 0:
            if len(module_output_shape) > 1:
                module_output_shape[0] -= 2 * self._remove_border
                module_output_shape[1] -= 2 * self._remove_border
                if len(module_output_shape) > 2:
                    module_output_shape[2] -= 2 * self._remove_border
        return module, module_output_shape, images, markers

    def _build_pool_layer(self, images, markers, batch_size, layer_config):
        device = self.device
        f_operations = {
            "max_pool2d": F.max_pool2d,
            "max_pool3d": F.max_pool3d,
            "avg_pool2d": F.avg_pool2d,
            "avg_pool3d": F.avg_pool3d,
        }
        operation_name = layer_config["operation"]
        operation_params = layer_config["params"]
        operation = __operations__[operation_name]
        f_pool = f_operations[operation_name]

        is_3d = "3d" in operation_name

        stride = operation_params.get("stride", 1)
        kernel_size = operation_params["kernel_size"]

        if "padding" in operation_params:
            padding = operation_params["padding"]
            if isinstance(padding, int):
                padding = [padding] * (3 if is_3d else 2)
        else:
            padding = [0] * (3 if is_3d else 2)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * (3 if is_3d else 2)

        operation_params["stride"] = 1
        # TODO what about dilation?
        # operation_params["padding"] = [k_size // 2 for k_size in kernel_size]
        if images is not None and markers is not None:
            torch_images = torch.from_numpy(images)

            if is_3d:
                torch_images = torch_images.permute(0, 4, 3, 1, 2)
            else:
                torch_images = torch_images.permute(0, 3, 1, 2)

            input_shape = torch_images.shape
            input_size = input_shape[0]

            outputs = torch.Tensor([])

            # temporarly ignore warnings till pytorch is fixed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with torch.no_grad():
                    for i in range(0, input_size, batch_size):
                        batch = torch_images[i : i + batch_size]
                        batch = batch.to(device)
                        output = f_pool(batch, **operation_params)
                        output = output.detach().cpu()
                        outputs = torch.cat((outputs, output))

            if is_3d:
                images = (
                    outputs.permute(0, 3, 4, 2, 1)
                    .detach()
                    .numpy()[:, :, : input_shape[2], : input_shape[3], : input_shape[3]]
                )
            else:
                images = (
                    outputs.permute(0, 2, 3, 1)
                    .detach()
                    .numpy()[:, :, : input_shape[2], : input_shape[3]]
                )

        operation_params["stride"] = stride
        operation_params["padding"] = padding
        layer = operation(**operation_params)
        return layer, images

    def _build_m_norm_layer(self, images, markers, input_shape, layer_config):
        operation_name = layer_config["operation"]
        operation_params = layer_config["params"]

        operation = __operations__[operation_name]

        is_3d = "3d" in layer_config["operation"]

        in_channels = input_shape[-1]
        if images is None or markers is None:
            mean = None
            std = None
            epsilon = 0.001
        else:
            kernel_size = 1
            dilation = 1
            epsilon = operation_params.get("epsilon", 0.001)

            if isinstance(dilation, int):
                dilation = [dilation] * (3 if is_3d else 2)

            if isinstance(kernel_size, int):
                kernel_size = [kernel_size] * (3 if is_3d else 2)

            patches, _ = _generate_patches(
                images, markers, in_channels, kernel_size, dilation
            )

            if is_3d:
                axis = (0, 1, 2, 3)
            else:
                axis = (0, 1, 2)

            mean = (
                torch.from_numpy(patches.mean(axis=axis, keepdims=True))
                .flatten()
                .float()
            )
            std = (
                torch.from_numpy(patches.std(axis=axis, keepdims=True))
                .flatten()
                .float()
            )

        layer = operation(mean=mean, std=std, in_channels=in_channels, epsilon=epsilon)

        return layer

    def _build_conv_layer(
        self,
        images,
        markers,
        remove_similar_filters,
        similarity_level,
        input_shape,
        layer_config,
    ):

        operation_name = layer_config["operation"]
        operation_params = layer_config["params"]
        operation = __operations__[operation_name]
        is_3d = operation_name == "conv3d"

        number_of_kernels_per_marker = operation_params.get(
            "number_of_kernels_per_marker", None
        )
        use_random_kernels = operation_params.get("use_random_kernels", False)

        use_pca = operation_params.get("use_pca", True)

        kernel_size = operation_params["kernel_size"]
        stride = operation_params.get("stride", 1)
        # TODO check padding is enough to maintain the input size
        padding = operation_params.get("padding", 0)
        padding_mode = operation_params.get("padding_mode", "zeros")
        dilation = operation_params.get("dilation", 1)
        groups = operation_params.get("groups", 1)
        bias = operation_params.get("bias", False)

        out_channels = operation_params.get("out_channels", None)
        # training parameters
        epochs = operation_params.get("epochs", 50)
        lr = operation_params.get("lr", 0.001)
        wd = operation_params.get("wd", 0.9)

        assert (
            out_channels is not None or markers is not None
        ), "`out_channels` or `markers` must be defined."

        in_channels = input_shape[-1]
        if isinstance(dilation, int):
            dilation = [dilation] * (3 if is_3d else 2)

        if isinstance(padding, int):
            padding = [padding] * (3 if is_3d else 2)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * (3 if is_3d else 2)

        if (
            markers is not None
            and "number_of_kernels_per_marker" not in operation_params
        ):
            number_of_kernels_per_marker = math.ceil(
                operation_params["out_channels"] / np.array(markers).max()
            )

        if out_channels is not None:
            assert out_channels is not None or (
                number_of_kernels_per_marker * np.array(markers).max() >= out_channels
            ), f"The number of kernels per marker is not enough to generate {out_channels} kernels."

        weights, bias_weights = _initialize_convNd_weights(
            images,
            markers,
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
            number_of_kernels_per_marker=number_of_kernels_per_marker,
            use_random_kernels=use_random_kernels,
            use_pca=use_pca,
            epochs=epochs,
            lr=lr,
            wd=wd,
            multi_level_clustering=self._multilevel_clustering,
            device=self.device,
        )
        if out_channels is not None:
            assert (
                weights.shape[0] >= out_channels
            ), f"Expected {out_channels} kernels and found {weights.shape[0]} kernels."
            if weights.shape[0] < out_channels:
                warnings.warn(
                    f"It was not possible to create {out_channels} kernels. It was created only {weights.shape[0]}."
                )

        if out_channels is None:
            out_channels = weights.shape[0]

        layer = operation(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        if images is not None and markers is not None:
            layer.weight = nn.Parameter(torch.from_numpy(weights))

            if bias:
                layer.bias = nn.Parameter(torch.from_numpy(bias_weights))

        else:
            nn.init.xavier_normal_(layer.weight)
            if bias:
                nn.init.zeros_(layer.bias)

        if remove_similar_filters:
            layer = _remove_similar_filters(layer, similarity_level)

        return layer

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
        labels = m[indices] - 1 if max_label > 1 else m[indices]

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
    if "operation" not in params:
        raise AssertionError("Layer does not have an operation.")

    if "params" not in params:
        raise AssertionError("Layer does not have operation params.")


def _pooling_markers(markers, kernel_size, stride=1, padding=0):
    new_markers = []
    for marker in markers:
        indices_x, indices_y = np.where(marker != 0)

        marker_shape = [*marker.shape]

        marker_shape[0] = math.floor(
            (marker_shape[0] + 2 * padding[0] - kernel_size[0]) / stride + 1
        )
        marker_shape[1] = math.floor(
            (marker_shape[1] + 2 * padding[1] - kernel_size[1]) / stride + 1
        )

        new_marker = np.zeros(marker_shape, dtype=np.int)
        x_limit = marker.shape[0] + 2 * padding[0] - kernel_size[0]
        y_limit = marker.shape[1] + 2 * padding[1] - kernel_size[1]
        for x, y in zip(indices_x, indices_y):
            if x > x_limit or y > y_limit:
                continue
            new_marker[x // stride][y // stride] = marker[x][y]

        new_markers.append(new_marker)

    return np.array(new_markers)


def _find_skip_connections_in_module(module_name, module):
    skips = dict()

    layers = module["layers"]

    for layer_name, layer_config in layers.items():
        key_name = (
            f"{module_name}.{layer_name}" if module_name is not None else layer_name
        )

        if "type" in layer_config:
            submodules_skips = _find_skip_connections_in_module(key_name, layer_config)

            skips.update(submodules_skips)

        if "inputs" in layer_config:
            skips[key_name] = layer_config["inputs"]

    return skips


def _find_skip_connections(arch):
    skips = dict()
    for module_name, module_arch in arch.items():
        sub_modules_skips = _find_skip_connections_in_module(module_name, module_arch)
        skips.update(sub_modules_skips)
    return skips


def _find_outputs_to_save(skips):
    outputs_to_save = {}
    for _, inputs in skips.items():

        for layer_name in inputs:
            outputs_to_save[layer_name] = True

    return outputs_to_save


def _create_random_kernels(n, in_channels, kernel_size):
    kernels = np.random.rand(n, in_channels, *kernel_size)

    return kernels


def _enforce_norm(kernels):
    kernels_shape = kernels.shape
    flattened_kernels = kernels.reshape(kernels_shape[0], -1)

    norm = np.linalg.norm(flattened_kernels, axis=1, keepdims=True)

    normalized = flattened_kernels / norm

    mean = normalized.mean(axis=1, keepdims=True)

    centered = normalized - mean

    centered = centered.reshape(*kernels_shape)

    return centered


def _create_random_pca_kernels(n, k, in_channels, kernel_size):

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * 2
    elif isinstance(kernel_size, list) and len(kernel_size) == 1:
        kernel_size = kernel_size * 2

    kernels = _enforce_norm(_create_random_kernels(n, in_channels, kernel_size))

    kernels_pca = _select_kernels_with_pca(kernels, k)

    return kernels_pca


def _select_kernels_with_pca(kernels, k, scale_kernels=False):
    kernels_shape = kernels.shape

    kernels_flatted = kernels.reshape(kernels_shape[0], -1)
    if k > kernels_flatted.shape[0] or k > kernels_flatted.shape[1]:
        k = min(kernels_flatted.shape[0], kernels_flatted.shape[1])

    pca = PCA(n_components=k)
    pca.fit(kernels_flatted)

    kernels_pca = pca.components_
    if scale_kernels:
        eigen_values = pca.singular_values_
        kernels_pca = kernels_pca * np.expand_dims(eigen_values, 1)
    kernels_pca = kernels_pca.reshape(-1, *kernels_shape[1:])

    return kernels_pca


def _generate_patches(images, markers, in_channels, kernel_size, dilation):
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
    kernel_size = np.array(kernel_size)
    dilation = np.array(dilation)

    dilated_kernel_size = kernel_size + (dilation - 1) * (kernel_size - 1)
    dilated_padding = dilated_kernel_size // 2

    is_2d = kernel_size.shape[0] == 2

    if is_2d:
        padding = (
            (dilated_padding[0], dilated_padding[0]),
            (dilated_padding[1], dilated_padding[1]),
            (0, 0),
        )
        patches_shape = (dilated_kernel_size[0], dilated_kernel_size[1], in_channels)
    else:
        padding = (
            (dilated_padding[0], dilated_padding[0]),
            (dilated_padding[1], dilated_padding[1]),
            (dilated_padding[2], dilated_padding[2]),
            (0, 0),
        )
        patches_shape = (
            dilated_kernel_size[0],
            dilated_kernel_size[1],
            dilated_kernel_size[2],
            in_channels,
        )

    all_patches, all_labels = None, None
    for image, image_markers in zip(images, markers):
        if len(image_markers) == 0:
            continue

        image_pad = np.pad(image, padding, mode="constant", constant_values=0)
        # TODO check if patches_shape is valid
        patches = view_as_windows(image_pad, patches_shape, step=1)

        shape = patches.shape
        image_shape = image.shape

        indices = np.where(image_markers != 0)
        markers_x = indices[0]
        markers_y = indices[1]
        if not is_2d:
            markers_z = indices[1]
        labels = image_markers[indices] - 1

        mask = np.logical_and(markers_x < image_shape[0], markers_y < image_shape[1])

        if not is_2d:
            mask = np.logical_and(mask, markers_z < image_shape[2])
            markers_z = markers_z[mask]

        markers_x = markers_x[mask]
        markers_y = markers_y[mask]

        labels = labels[mask]

        if is_2d:
            generated_patches = patches[markers_x, markers_y].reshape(-1, *shape[3:])
        else:
            generated_patches = patches[markers_x, markers_y, markers_z].reshape(
                -1, *shape[4:]
            )

        if is_2d:
            if dilation[0] > 1 or dilation[1] > 1:
                r = np.arange(0, dilated_kernel_size[0], dilation[0])
                s = np.arange(0, dilated_kernel_size[1], dilation[1])
                generated_patches = generated_patches[:, r, :, :][:, :, s, :]
        else:
            if dilation[0] > 1 or dilation[1] > 1 or dilation[2] > 1:
                r = np.arange(0, dilated_kernel_size[0], dilation[0])
                s = np.arange(0, dilated_kernel_size[1], dilation[1])
                t = np.arange(0, dilated_kernel_size[2], dilation[2])
                generated_patches = generated_patches[:, r, :, :, :][:, :, s, :, :][
                    :, :, :, t, :
                ]

        if all_patches is None:
            all_patches = generated_patches
            all_labels = labels
        else:
            all_patches = np.concatenate((all_patches, generated_patches))
            all_labels = np.concatenate((all_labels, labels))

    return all_patches, all_labels


def _points_closest_to_centers(points, centers):
    _points = []
    for center in centers:
        _center = np.expand_dims(center, 0)

        dist = distance.cdist(points, _center)

        _points.append(points[np.argmin(dist)])

    return np.array(_points)


def _find_elems_in_array(a, elems):
    indices = []
    for elem in elems:
        _elem = np.expand_dims(elem, 0)
        mask = np.all(a == _elem, axis=1)

        indice = np.where(mask)[0][0:1].item()

        indices.append(indice)

    return indices


def _kmeans_roots(
    patches,
    labels,
    n_clusters_per_label,
    return_labels=False,
    random_state=None,
    distance_metric="euclidean",
):
    """Cluster patch and return the root of each custer.

    Parameters
    ----------
    patches : ndarray
        Array of patches with shape :math:`((N, H, W, C))`
    labels : ndarray
        The label of each patch with shape :nath:`(N,)`
    n_clusters_per_label : int
        The number os clusters per label.
    compute_bias : bool, optional
        If True, compute the bias of each cluster, by default False.

    Returns
    -------
    ndarray
        A array with all the roots.

    """
    roots = None
    min_number_of_pacthes_per_label = n_clusters_per_label

    possible_labels = np.unique(labels)
    cluster_labels = np.zeros_like(labels)
    last_label = 0
    for i, label in enumerate(possible_labels):
        patches_of_label = patches[label == labels].astype(np.float32)
        # TODO get a value as arg.
        if patches_of_label.shape[0] > min_number_of_pacthes_per_label:
            # TODO remove fix random_state
            # kmeans = MiniBatchKMeans(
            #    n_clusters=n_clusters_per_label, max_iter=300, random_state=42, init_size=3 * n_clusters_per_label)

            kmeans = KMeans(
                n_clusters=n_clusters_per_label,
                max_iter=100,
                tol=0.001,
                random_state=42,
                n_init=10,
            )

            if distance_metric == "cosine":

                def _metric(x, y):
                    return 1.0 - cosine_similarity(x, y)

                kmeans.euclidean_distances = _metric
                kmeans._euclidean_distances = _metric

            kmeans.fit(patches_of_label.reshape(patches_of_label.shape[0], -1))
            centers = kmeans.cluster_centers_
            current_labels = kmeans.labels_ + last_label
            cluster_labels[label == labels] = current_labels
            last_label = current_labels.max() + 1

            kmeans.euclidean_distances = euclidean_distances
            kmeans._euclidean_distances = _euclidean_distances

            # roots_of_label = _points_closest_to_centers(
            #    patches_of_label.reshape(patches_of_label.shape[0], -1), centers
            # )
            roots_of_label = centers

        # TODO is enough to check if is equal?
        else:
            roots_of_label = patches_of_label.reshape(patches_of_label.shape[0], -1)
            current_labels = np.arange(roots_of_label.shape[0]) + last_label
            cluster_labels[label == labels] = current_labels
            last_label = current_labels.max() + 1

        if roots is not None:
            roots = np.concatenate((roots, roots_of_label))
        else:
            roots = roots_of_label

    roots = roots.reshape(-1, *patches.shape[1:])

    if return_labels:
        return roots, cluster_labels

    return roots


def _calculate_convNd_weights(
    images,
    markers,
    in_channels,
    kernel_size,
    dilation,
    bias,
    number_of_kernels_per_marker,
    num_kernels,
    epochs,
    lr,
    wd,
    use_pca,
    multilevel_clustering,
    device="cpu",
):
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
            and the third row is the markers pixel labels

        Returns
        -------
        ndarray
            Kernels weights in the shape \
            :math:`(N \times C \times H \times W)`.
        
        """
    patches, labels = _generate_patches(
        images, markers, in_channels, kernel_size, dilation
    )

    axis = tuple(range(len(kernel_size) + 1))

    # TODO is it needed?
    # mean_by_channel = patches.mean(axis=axis, keepdims=True)
    # std_by_channel = patches.std(axis=axis, keepdims=True)
    #
    # patches = (patches - mean_by_channel)/(std_by_channel + default_std)

    # force norm 1
    # patches_shape = patches.shape
    # patches = patches.reshape(patches_shape[0], -1)
    # patches = patches / (np.linalg.norm(patches, axis=1, keepdims=True) + 1e-6)
    # patches = patches.reshape(patches_shape)

    if bias:
        kernel_weights, bias_weights = _compute_kernels_with_backpropagation(
            patches,
            labels,
            number_of_kernels_per_marker,
            num_kernels,
            epochs,
            lr,
            wd,
            use_pca,
            multilevel_clustering,
            device,
        )
    else:
        kernel_weights = _kmeans_roots(patches, labels, number_of_kernels_per_marker)

        # force norm 1
        kernel_weights = force_norm_1(kernel_weights)

    if bias:
        return kernel_weights, bias_weights
    else:
        return kernel_weights


def force_norm_1(kernel_weights):
    kernel_weights_shape = kernel_weights.shape
    kernel_weights = kernel_weights.reshape(kernel_weights_shape[0], -1)
    kernel_weights = kernel_weights / (
        np.linalg.norm(kernel_weights, axis=1, keepdims=True) + DIVISION_EPSILON
    )
    kernel_weights = kernel_weights.reshape(kernel_weights_shape)
    return kernel_weights


def _compute_kernels_with_backpropagation(
    patches,
    patches_labels,
    num_kernels_per_marker,
    num_kernels,
    epochs=50,
    lr=0.001,
    wd=0.9,
    use_pca=True,
    multi_level_clustering=True,
    device="cpu",
):
    patches_shape = patches.shape
    patches = patches.reshape(patches_shape[0], -1)

    # cluster patches
    if multi_level_clustering:
        cluster_centers, labels = _kmeans_roots(
            patches, patches_labels, num_kernels_per_marker, return_labels=True
        )

        new_patche_labels = np.zeros(patches_labels.shape, dtype=np.int64)

        if use_pca and np.prod(cluster_centers.shape[1:]) < cluster_centers.shape[0]:
            kernel_shape = patches_shape[1:]
            # kernels = cluster_centers.reshape(-1, *kernel_shape)
            kernels = cluster_centers
            kernels = kernels.reshape(-1, *kernel_shape)
            kernels = force_norm_1(kernels)
            kernels = _kernels_to_channel_first(kernels)
            kernels = _select_kernels_with_pca(kernels, num_kernels)
            bias = np.zeros(num_kernels, dtype=np.float32)
            kernels = _kernels_to_channel_last(kernels)
            # TODO do not return form here
            return kernels, bias

        if num_kernels < cluster_centers.shape[0]:
            new_cluster_centers, new_labels = _kmeans_roots(
                cluster_centers,
                np.ones(cluster_centers.shape[0], dtype=np.int64),
                num_kernels,
                return_labels=True,
                distance_metric="cosine",
            )

            for i, new_label in enumerate(new_labels):
                label = i
                mask = labels == label
                new_patche_labels[mask] = new_label

        else:
            new_cluster_centers = cluster_centers
            new_patche_labels = labels

        labels = new_patche_labels
        init_kernels = new_cluster_centers

    else:
        kmeans = KMeans(
            n_clusters=num_kernels, max_iter=100, tol=0.001, random_state=42
        )
        kmeans.fit(patches)
        labels = kmeans.labels_
        init_kernels = kmeans.cluster_centers_

    lin_layer = nn.Linear(patches.shape[1], num_kernels, bias=True).to(device)
    act_layer = nn.ReLU(True).to(device)
    # nn.init.xavier_normal_(lin_layer.weight, gain=nn.init.calculate_gain("relu"))
    init_kernels = init_kernels / (
        np.linalg.norm(init_kernels, axis=1, keepdims=True) + DIVISION_EPSILON
    )
    lin_layer.weight.data = torch.from_numpy(init_kernels).to(device)
    nn.init.constant_(lin_layer.bias, 0)

    criterion = CrossEntropyLoss()

    inputs = torch.from_numpy(patches).float().to(device)
    true_labels = torch.from_numpy(labels).long().to(device)

    optim = torch.optim.Adam(lin_layer.parameters(), lr=lr, weight_decay=wd)
    print("Building layer...")
    for epoch in range(epochs):
        loss_epoch = 0.0

        optim.zero_grad()
        outputs = act_layer(lin_layer(inputs))
        loss = criterion(outputs, true_labels)
        loss.backward()
        preds = torch.argmax(outputs, dim=1)

        # gradient clip
        # nn.utils.clip_grad_norm_(lin_layer.parameters(), 0.1)
        optim.step()

        loss_epoch = loss.item()

        acc = np.mean(preds.detach().cpu().numpy() == labels)
        print("Epoch {}: loss = {}, accuracy = {}".format(epoch, loss_epoch, acc))

        if loss_epoch < 0.01:
            break

    kernels = lin_layer.weight.detach().cpu().numpy()
    bias = lin_layer.bias.detach().cpu().numpy()
    kernels = force_norm_1(kernels)
    kernels = kernels.reshape(-1, *patches_shape[1:])
    return kernels, bias


def _initialize_convNd_weights(
    images=None,
    markers=None,
    in_channels=3,
    out_channels=None,
    kernel_size=None,
    dilation=None,
    bias=False,
    number_of_kernels_per_marker=16,
    use_random_kernels=False,
    use_pca=True,
    epochs=50,
    lr=0.001,
    wd=0.9,
    multi_level_clustering=True,
    device="cpu",
):
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

    bias_weights = None

    if use_random_kernels:
        kernels_weights = _create_random_pca_kernels(
            n=out_channels * 10,
            k=out_channels,
            in_channels=in_channels,
            kernel_size=kernel_size,
        )

    elif images is not None and markers is not None:

        if bias:
            markers = markers.copy()
            # markers[markers != 0] = 1
            # number_of_kernels_per_marker = out_channels

        weights = _calculate_convNd_weights(
            images,
            markers,
            in_channels,
            kernel_size,
            dilation,
            bias,
            number_of_kernels_per_marker,
            out_channels,
            epochs=epochs,
            lr=lr,
            wd=wd,
            use_pca=use_pca,
            multilevel_clustering=multi_level_clustering,
            device=device,
        )
        if bias:
            kernels_weights, bias_weights = weights
        else:
            kernels_weights = weights

        kernels_weights = _kernels_to_channel_first(kernels_weights)

        assert (
            out_channels is None or kernels_weights.shape[0] >= out_channels
        ), "Not enough kernels were generated!!!"

        if not bias:
            if (
                use_pca
                and (out_channels is not None)
                and (np.prod(kernels_weights.shape[1:]) < kernels_weights.shape[0])
            ):
                kernels_weights = _select_kernels_with_pca(
                    kernels_weights, out_channels
                )

            elif (out_channels is not None) and (
                out_channels < kernels_weights.shape[0]
            ):
                kernels_weights = _kmeans_roots(
                    kernels_weights,
                    np.ones(kernels_weights.shape[0]),
                    out_channels,
                    distance_metric="cosine",
                )

    else:
        kernels_weights = torch.rand(out_channels, in_channels, *kernel_size).numpy()

        bias_weights = np.zeros(out_channels)

    kernels_weights = force_norm_1(kernels_weights)

    return kernels_weights, bias_weights


def _kernels_to_channel_first(kernels_weights):
    if kernels_weights.ndim == 4:
        kernels_weights = kernels_weights.transpose(0, 3, 1, 2)
    else:
        kernels_weights = kernels_weights.transpose(0, 4, 3, 1, 2)
    return kernels_weights


def _kernels_to_channel_last(kernels_weights):
    if kernels_weights.ndim == 4:
        kernels_weights = kernels_weights.transpose(0, 2, 3, 1)
    else:
        kernels_weights = kernels_weights.transpose(0, 2, 4, 3, 1)
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
    assert 0 < similarity_level <= 1, "Similarity must be in range (0, 1]"

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

    new_conv = nn.Conv2d(
        layer.in_channels,
        out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        bias=layer.bias,
        padding=layer.padding,
    )

    new_conv.weight = nn.Parameter(selected_filters)

    return new_conv
