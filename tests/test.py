import unittest
from collections import OrderedDict
from math import ceil
from os import path
from unittest import TestCase

import torch

from flim.experiments import utils as flim_utils
from flim.models.lcn import LCNCreator, LIDSConvNet

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class TestBuildSimpleModel(TestCase):
    def test_build_model(self):
        arch = flim_utils.load_architecture(
            path.join(path.dirname(__file__), "data", "arch.json")
        )

        creator = LCNCreator(
            architecture=arch, input_shape=[224, 224, 3], device=device
        )
        creator.build_model()
        model = creator.get_LIDSConvNet().to(device)

        self.assertIsInstance(model, LIDSConvNet)

        self.assertIsNotNone(model.features)
        self.assertIsNotNone(model.classifier)

        fake_input = torch.randn(1, 3, 224, 224).to(device)
        outputs = OrderedDict()

        # get output of each module
        outputs["features"] = model.features(fake_input)
        outputs["classifier"] = model.classifier(outputs["features"].flatten(1))

        # check if output of each module is correct
        self.assertEqual(outputs["features"].shape, torch.Size([1, 27, 56, 56]))
        self.assertEqual(outputs["classifier"].shape, torch.Size([1, 2]))


class TestBuildModelWithMarkers(TestCase):
    def test_build_model(self):
        print(path.dirname(__file__))
        arch = flim_utils.load_architecture(
            path.join(path.dirname(__file__), "data", "arch.json")
        )

        images, markers = flim_utils.load_images_and_markers(
            path.join(path.dirname(__file__), "data", "images_and_markers")
        )

        creator = LCNCreator(
            architecture=arch,
            images=images,
            markers=markers,
            device=device,
        )

        creator.build_model()
        model = creator.get_LIDSConvNet().to(device)

        self.assertIsInstance(model, LIDSConvNet)

        self.assertIsNotNone(model.features)
        self.assertIsNotNone(model.classifier)

        torch_input = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)
        outputs = OrderedDict()

        # get output of each module
        for name, module in model.named_children():
            outputs[name] = module(torch_input)
            torch_input = outputs[name]

        # check if output of each module is correct
        self.assertEqual(
            outputs["features"].shape,
            torch.Size([1, 27, ceil(images.shape[1] / 4), ceil(images.shape[2] / 4)]),
        )
        self.assertEqual(outputs["classifier"].shape, torch.Size([1, 2]))


class TestParallelModule(TestCase):
    _arch = {
        "module": {
            "type": "parallel",
            "params": {"aggregate": "concat"},
            "layers": {
                "conv1": {
                    "operation": "conv2d",
                    "params": {"out_channels": 32, "kernel_size": 3, "padding": 1},
                },
                "conv2": {
                    "operation": "conv2d",
                    "params": {"out_channels": 16, "kernel_size": 5, "padding": 2},
                },
                "conv3": {
                    "operation": "conv2d",
                    "params": {
                        "out_channels": 4,
                        "kernel_size": 5,
                        "dilation": 3,
                        "padding": 6,
                    },
                },
            },
        }
    }

    def test_parallel_module_concat(self):
        arch = self._arch.copy()

        creator = LCNCreator(architecture=arch, input_shape=[3], device=device)
        creator.build_model()
        model = creator.get_LIDSConvNet().to(device)

        fake_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(fake_input)

        self.assertEqual(output.shape, torch.Size([1, 52, 224, 224]))

    def test_parallel_module_sum(self):
        arch = self._arch.copy()
        arch["module"]["params"]["aggregate"] = "sum"
        arch["module"]["layers"]["conv1"]["params"]["out_channels"] = 16
        arch["module"]["layers"]["conv2"]["params"]["out_channels"] = 16
        arch["module"]["layers"]["conv3"]["params"]["out_channels"] = 16

        creator = LCNCreator(architecture=arch, input_shape=[3], device=device)
        creator.build_model()
        model = creator.get_LIDSConvNet().to(device)

        fake_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(fake_input)

        self.assertEqual(output.shape, torch.Size([1, 16, 224, 224]))

    def test_parallel_module_prod(self):
        arch = self._arch.copy()
        arch["module"]["params"]["aggregate"] = "prod"
        arch["module"]["layers"]["conv1"]["params"]["out_channels"] = 16
        arch["module"]["layers"]["conv2"]["params"]["out_channels"] = 16
        arch["module"]["layers"]["conv3"]["params"]["out_channels"] = 16

        creator = LCNCreator(architecture=arch, input_shape=[3], device=device)
        creator.build_model()
        model = creator.get_LIDSConvNet().to(device)

        fake_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(fake_input)

        self.assertEqual(output.shape, torch.Size([1, 16, 224, 224]))


class TestConvWithBias(TestCase):
    def test_conv_with_bias(self):
        arch = {
            "module": {
                "type": "sequential",
                "flim_params": {
                    "epochs": 2,
                    "lr": 0.001,
                    "wd": 0.9,
                },
                "layers": {
                    "conv": {
                        "operation": "conv2d",
                        "params": {
                            "kernel_size": 5,
                            "stride": 1,
                            "padding": 2,
                            "dilation": 1,
                            "out_channels": 64,
                            "bias": True,
                        },
                    }
                },
            }
        }

        images, markers = flim_utils.load_images_and_markers(
            path.join(path.dirname(__file__), "data", "images_and_markers")
        )
        creator = LCNCreator(
            architecture=arch,
            images=images,
            markers=markers,
            device=device,
            relabel_markers=False,
        )

        creator.build_model()
        model = creator.get_LIDSConvNet().to(device)

        torch_images = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)
        output = model(torch_images)

        self.assertEqual(
            output.shape, torch.Size([1, 64, images.shape[1], images.shape[2]])
        )
        self.assertIsNotNone(model.module.conv.bias)


if __name__ == "__main__":
    unittest.main()
