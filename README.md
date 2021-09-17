# FLIM - Feature Learning from Image Markers

Feature Learning from Image Markers is a technique to learn the filter of convolutional neural networks' feature extractors from user-drawn image markers.

This package provides an implementation of this technique powered by Pytorch, Scikit-learn, and Scikit-image.

 To install dependencies run the command

 ```
 pip install -r requirements.txt
 ```

To install it, go to the folder where the package it and run the command

 ```
 pip install . 
 ```

> :warning: It is recommended to create a virtual enviroment before instaling this package.

To use NVIDIA GPUs, install the PyTorch version for your specific CUDA version. Instructions can be found [here](https://pytorch.org/get-started/locally/?source=Google&medium=PaidSearch&utm_campaign=1712411904&utm_adgroup=66400476185&utm_keyword=install%20pytorch&utm_offering=AI&utm_Product=PYTorch&gclid=CjwKCAjwh7H7BRBBEiwAPXjaduQvhmeLJWAM3I-IXfEKzXHKkvyD7goKEVfInMqa845hvyoOY6AcoBoCyHMQAvD_BwE).

To build the package API reference documentation, go to the folder `docs` and run the command

```
make html
```

You can run a simple HTTP server to serve the documetation page.

```
python -m http.server
```

Go to [localhost:8000](localhost:8000) and you navigate through the documentation.

## How to define the model architecture with JSON

It is possible to define the model architecture using a JSON that must respect the following structure:

```
{
    "features": ...
    "classifier": ...
}
```

The key "features" specifies the feature extractor architecture, and the key "classifier" specifies a classifier. The key "classifier" is optional.

An example of a feature extractor:

```
"features": {
        "type": "sequential",
        "layers": {
            "conv1": {
                "operation": "conv2d",
                "params": {
                    "kernel_size": 5,
                    "stride": 1,
                    "padding": 2,
                    "dilation": 1,
                    "number_of_kernels_per_marker": 8
                }
            },
            "activation": {
                "operation": "relu",
                "params": {
                    "inplace": true
                }
            },

            "pool": {
                "operation": "max_pool2d",
                "params": {
                    "kernel_size": 3,
                    "stride": 4,
                    "padding": 0
                }
            }
        }
    }
```

In this example, "features" is a module of type `sequential` - that is, each sub-module is applied in the order that is specified. This "features" module has a field called "layers" that determines its layers. Layers can be modules.

Each layer has a name, which is the key, and it is necessary to inform the type of the layer through the "operation" field. A list of currently supported operations:

* "max_pool2d"
* "conv2d"
* "relu"
* "linear"
* "batch_norm2d"
* "dropout"
* "adap_avg_pool2d"
* "unfold"
* "fold"

Every layer must be specified as follows:

```
"layer_name": {
    "operation": "operation_name",
    "params: {
        any necessary parameter to create this layer
    }
}
``` 
To know which parameters for each type of operation, just look at the [PyTorch documentation](https://pytorch.org/docs/stable/nn.html). All parameters specified there are supported with the same name.

A layer with the "conv2d" operation accepts a few more fields: "activation" and "pool". These fields can be used to define an activation function and pooling operation, but these operations can be specified outside the convolucioal layer as well. A convolutional layer also has a "number_of_kernels_per_marker" parameter that determines the number of kernels that must be created from the markers.

An example of a classifier follows:

```
"classifier": {
    "layers": {
        "linear1": {
            "operation": "linear",
            "params": {
                "in_features": -1,
                "out_features": 512
            }
        },
       
        "relu3": {
            "operation": "relu",
            "params": {
                "inplace": true
            }
        },

        "linear2": {
            "operation": "linear",
            "params": {
                "in_features": 512,
                "out_features": 512
            }
        }
    }
}
```

Layers with "linear" operation (fully connected) with the parameter "in_feature" as -1 have this information determined automatically. The number of features may depend on the feature extractor (number of kernels, padding, stride, etc.).

## CLI Tools

When the package is installed with `pip`, there are two command line tools that are available for use in any directory.

The available tools are `train` and` validate`. For a description of the parameters, run `train -h` or` validate -h` on the console.

Following, there are usage examples for each tool:

```
train train -d lids-dataset -ts train_split.txt -ad arch.json -md markers-dir -od outputs-dir -mn trained-model.pt -g 0 -e 120 -bs 64 -lr 0.001 -wd 0.001
```

```
validate -d lids-dataset -vs val_split.txt -ad arch.json -mp outputs-dir/trained-model.pt -g 0
```

To save the output of intermediate layers, run:

```
train train -md markers/ -d lids-dataset/ -ts train.txt -ad arch.json -g 0 -od outputs -mn model.pt -i -f zip -l layer_name1 layer_name2
```

The possible formats are OPFDataset (zip), MImage (mimg), and NumPy array (npy).

The dataset must be in LIDS format. It is also possible to train an SVM model by running `train` passing the `-s` argument.

Please, feel free to give feedback and to contribute. If you have any question, you can open an issue requesting help or you can contact me.