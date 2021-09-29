from collections import OrderedDict
import unittest
from unittest import TestCase

from os import path
from math import ceil

import torch
from torch.nn.modules.container import T

from flim.models.lcn import LCNCreator, LIDSConvNet
from flim.experiments import utils as flim_utils

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class TestBuildSimpleModel(TestCase):
    def test_build_model(self):
        print(path.dirname(__file__))
        arch = flim_utils.load_architecture(path.join(path.dirname(__file__), 'data', 'arch.json'))

        creator = LCNCreator(architecture=arch, input_shape=[3], device=device)
        creator.build_model()
        model = creator.get_LIDSConvNet().to(device)

        self.assertIsInstance(model, LIDSConvNet)

        self.assertIsNotNone(model.features)
        self.assertIsNotNone(model.classifier)       

        fake_input = torch.randn(1, 3, 224, 224).to(device)
        outputs = OrderedDict()

        # get output of each module
        for name, module in model.named_children():
            outputs[name] = module(fake_input)
            fake_input = outputs[name]

        # check if output of each module is correct
        self.assertEqual(outputs['features'].shape, torch.Size([1, 64, 56, 56]))
        self.assertEqual(outputs['classifier'].shape, torch.Size([1, 2]))

class TestBuildModelWithMarkers(TestCase):
    def test_build_model(self):
        print(path.dirname(__file__))
        arch = flim_utils.load_architecture(path.join(path.dirname(__file__), 'data', 'arch.json'))

        images, markers = flim_utils.load_images_and_markers(path.join(path.dirname(__file__), 'data', 'images_and_markers'))

        creator = LCNCreator(architecture=arch,
                             images=images,
                             markers=markers,
                             device=device)

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
        self.assertEqual(outputs['features'].shape, torch.Size([1, 64, ceil(images.shape[1]/4), ceil(images.shape[2]/4)]))
        self.assertEqual(outputs['classifier'].shape, torch.Size([1, 2]))

class TestParallelModule(TestCase):
    _arch = {
            "module": {
                "type": "parallel",
                "aggregate": "concat",
                "layers": {
                    "conv1": {
                        "operation": "conv2d",
                        "params": {
                            "out_channels": 32,
                            "kernel_size": 3,
                            "padding": 1
                        }
                    },
                    "conv2": {
                        "operation": "conv2d",
                        "params": {
                            "out_channels": 16,
                            "kernel_size": 5,
                            "padding": 2
                        }
                    },
                    "conv3": {
                        "operation": "conv2d",
                        "params": {
                            "out_channels": 4,
                            "kernel_size": 5,
                            "dilation": 3,
                            "padding": 6
                        }
                    }
                }
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
        arch["module"]["aggregate"] = "sum"
        arch["module"]["layers"]["conv1"]["out_channels"] = 16
        arch["module"]["layers"]["conv2"]["out_channels"] = 16
        arch["module"]["layers"]["conv3"]["out_channels"] = 16

        creator = LCNCreator(architecture=arch, input_shape=[3], device=device)
        creator.build_model()
        model = creator.get_LIDSConvNet().to(device)

        fake_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(fake_input)

        self.assertEqual(output.shape, torch.Size([1, 16, 224, 224]))

    def test_parallel_module_prod(self):
        arch = self._arch.copy()
        arch["module"]["aggregate"] = "prod"
        arch["module"]["layers"]["conv1"]["out_channels"] = 16
        arch["module"]["layers"]["conv2"]["out_channels"] = 16
        arch["module"]["layers"]["conv3"]["out_channels"] = 16

        creator = LCNCreator(architecture=arch, input_shape=[3], device=device)
        creator.build_model()
        model = creator.get_LIDSConvNet().to(device)

        fake_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(fake_input)

        self.assertEqual(output.shape, torch.Size([1, 16, 224, 224]))

if __name__ == '__main__':
    unittest.main()