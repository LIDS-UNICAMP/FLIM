import torch
import torch.nn as nn

import logging

class LIDSConvNet(nn.Sequential):
    def __init__(self):
        super(LIDSConvNet, self).__init__()
        
        self.feature_extractor = nn.Sequential()

        self.logger = logging.getLogger()
        
    def forward(self, X, return_layeerwise_output=False):
        self.logger.info("doing forward")

        if return_layeerwise_output:
            output_by_layer = {}
        for layer_name, layer in self.feature_extractor.named_children():
            y = layer.forward(X)
            X = y

            if return_layeerwise_output:
                output_by_layer[layer_name] = y.detach().cpu()

        if return_layeerwise_output:
            return y, output_by_layer
        
        return y

    def to(self, device):
        for _, layer in self.feature_extractor.named_children():
           layer.to(device)
        
        return self