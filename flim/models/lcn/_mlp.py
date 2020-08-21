import torch

import torch.nn as nn

operations = {
    "max_pool2d": nn.MaxPool2d,
    "relu": nn.ReLU,
    "linear": nn.Linear,
    "batch_norm2d": nn.BatchNorm2d,
    "dropout": nn.Dropout,
    "linear": nn.Linear,
    "adap_avg_pool2d": nn.AdaptiveAvgPool2d,
    "unfold": nn.Unfold,
    "fold": nn.Fold
}

class MLP(nn.Sequential):
    
    def __init__(self, architecture, output_shape):
        super(MLP, self).__init__()
        
        self._architecture = architecture
        self._output_shape = output_shape
        self._build()
        
    def forward(self, x):
        # print("there is a nan:", torch.isnan(x).any())
        # input_shape = x.size()
        # x = x.reshape(input_shape[0]*input_shape[1], input_shape[2])
        
        for name, layer in self.named_children():
            if isinstance(layer, nn.Fold):
                # x.reshape(input_shape[0], input_shape[1], -1).permute(0, 2, 1)
                x = x.permute(0, 2, 1)

            x = layer.forward(x)
            
            if isinstance(layer, nn.Unfold):
                x = x.permute(0, 2, 1)

        return x
    
    def to(self, device):
        for _, layer in self.named_children():
           layer.to(device)
        
        return self
    
    def _build(self):
        
        architecture = self._architecture['layers']
        
        print(architecture)
        
        for key in architecture:
            layer_config = architecture[key]
            
            operation = operations[layer_config['operation']]
            if layer_config['operation'] == "fold":
                layer_config['params']['output_size'] = self._output_shape
            operation_params = layer_config['params']
            
            layer = operation(**operation_params)
            
            self.add_module(key, layer)
            
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)