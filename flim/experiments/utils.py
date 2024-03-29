import json
import math
import os
import re
import warnings
from collections import OrderedDict
from math import floor

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from termcolor import colored
from torch.utils.data import DataLoader
from torchvision.models import vgg16_bn
from torchvision.transforms import Resize
from torch.nn.modules.conv import Conv3d
from torch.nn.modules.conv import Conv2d

from ..models.lcn import (
    LCNCreator,
    LIDSConvNet,
    MarkerBasedNorm2d,
    MarkerBasedNorm3d,
    ParallelModule,
)
from ._dataset import LIDSDataset
from ._image_utils import *

try:
    import pyift.pyift as ift
except ModuleNotFoundError:
    ift = None
    warnings.warn("PyIFT is not installed.", ImportWarning)


def load_markers(markers_dir):
    markers = []
    lines = None
    with open(markers_dir, "r") as f:
        lines = f.readlines()

    label_infos = [int(info) for info in lines[0].split(" ")]

    is_2d = len(label_infos) == 3

    if is_2d:
        image_shape = (label_infos[2], label_infos[1])
    else:
        image_shape = (label_infos[2], label_infos[1], label_infos[3])

    markers = np.zeros(image_shape, dtype=int)

    for line in lines[1:]:
        split_line = line.split(" ")
        if is_2d:
            y, x, label = int(split_line[0]), int(split_line[1]), int(split_line[3])
            markers[x][y] = label
        else:
            y, x, z, label = (
                int(split_line[0]),
                int(split_line[1]),
                int(split_line[2]),
                int(split_line[4]),
            )
            markers[x][y][z] = label

    return markers


def load_images_and_markers(path, normalize=True):
    dirs = os.listdir(path)
    images_names = [filename for filename in dirs if not filename.endswith(".txt")]
    makers_names = [filename for filename in dirs if filename.endswith(".txt")]

    images_names.sort()
    makers_names.sort()

    images = []
    images_markers = []

    for image_name, marker_name in zip(images_names, makers_names):

        if image_name.endswith(".npy"):
            image = np.load(os.path.join(path, image_name))
        else:
            image = load_image(os.path.join(path, image_name))
        markers = load_markers(os.path.join(path, marker_name))

        images.append(image)
        images_markers.append(markers)

    images = np.array(images)
    # if normalize:
    #    images = images / images.max(axis=(3, 2, 1), keepdims=True)
    images_markers = np.array(images_markers)

    return images, images_markers


def _convert_arch_from_lids_format(arch):
    stdev_factor = arch["stdev_factor"]

    n_layers = arch["nlayers"]

    n_arch = {"type": "sequential", "layers": {}}

    for i in range(1, n_layers + 1):
        layer_name = f"layer{i}"
        layer_params = arch[layer_name]

        kernel_size = layer_params["conv"]["kernel_size"]

        is3d = kernel_size[2] > 0

        end = 3 if is3d else 2
        dilation_rate = layer_params["conv"]["dilation_rate"][:end]
        kernel_size = kernel_size[:end]

        m_norm_layer = {
            "operation": "m_norm3d" if is3d else "m_norm2d",
            "params": {
                "kernel_size": kernel_size,
                "dilation": dilation_rate,
                "default_std": stdev_factor,
            },
        }

        conv_layer = {
            "operation": "conv3d" if is3d else "conv2d",
            "params": {
                "kernel_size": kernel_size,
                "dilation": dilation_rate,
                "number_of_kernels_per_marker": layer_params["conv"][
                    "nkernels_per_image"
                ],
                "padding": [k_size // 2 for k_size in kernel_size],
                "out_channels": layer_params["conv"]["noutput_channels"],
                "stride": 1,
            },
        }

        relu_layer = None

        if layer_params["relu"]:
            relu_layer = {"operation": "relu", "params": {"inplace": True}}

        pool_type_mapping = {
            "max_pool2d": "max_pool2d",
            "avg_pool2d": "avg_pool2d",
            "max_pool3d": "max_pool3d",
            "avg_pool3d": "avg_pool3d",
            "no_pool": None,
        }

        pool_type = layer_params["pooling"]["type"]

        if is3d and pool_type != "no_pool":
            pool_type += "3d"

        elif pool_type != "no_pool":
            pool_type += "2d"

        assert (
            pool_type in pool_type_mapping
        ), f"{pool_type} is not a supported pooling operation"

        if pool_type == "no_pool":
            pool_layer = None
        else:
            pool_kernel_size = layer_params["pooling"]["size"][:end]
            pool_layer = {
                "operation": pool_type_mapping[pool_type],
                "params": {
                    "kernel_size": pool_kernel_size,
                    "stride": layer_params["pooling"]["stride"],
                    "padding": [k_size // 2 for k_size in pool_kernel_size],
                },
            }

        n_arch["layers"][f"m-norm{i}"] = m_norm_layer
        n_arch["layers"][f"conv{i}"] = conv_layer
        if relu_layer:
            n_arch["layers"][f"activation{i}"] = relu_layer
        if pool_layer:
            n_arch["layers"][f"pool{i}"] = pool_layer

    return {"features": n_arch}


def load_architecture(architecture_dir):
    path = architecture_dir
    with open(path) as json_file:
        architecture = json.load(json_file)
    if "nlayers" in architecture:
        architecture = _convert_arch_from_lids_format(architecture)
    return architecture


def configure_dataset(dataset_dir, split_dir, lab=True, transform=None):
    dataset = LIDSDataset(dataset_dir, split_dir, lab, transform)

    return dataset


def build_model(
    architecture,
    images=None,
    markers=None,
    input_shape=None,
    batch_size=32,
    train_set=None,
    remove_border=0,
    relabel_markers=True,
    device="cpu",
    verbose=False,
):

    creator = LCNCreator(
        architecture,
        images=images,
        markers=markers,
        input_shape=input_shape,
        batch_size=batch_size,
        relabel_markers=relabel_markers,
        remove_border=remove_border,
        device=device,
    )
    if verbose:
        print("Building model...")
    creator.build_model(verbose=verbose)

    model = creator.get_LIDSConvNet()

    if verbose:
        print("Model ready.")

    return model


def get_torchvision_model(model_name, number_classes, pretrained=True, device="cpu"):
    model = None
    if model_name == "vgg16_bn":
        if pretrained:
            model = vgg16_bn(pretrained=pretrained)
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, number_classes),
            )

            for m in model.classifier.modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

        else:
            model = vgg16_bn(num_classes=number_classes, init_weights=True)

        model.to(device)
    return model


def train_mlp(
    model,
    train_set,
    epochs=30,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-3,
    criterion=nn.CrossEntropyLoss(),
    device="cpu",
):

    dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model.to(device)
    model.feature_extractor.eval()
    model.classifier.train()

    # optimizer
    optimizer = optim.Adam(
        model.classifier.parameters(), lr=lr, weight_decay=weight_decay
    )

    # training
    print(f"Training classifier for {epochs} epochs")
    for epoch in range(0, epochs):
        print("-" * 40)
        print("Epoch {}/{}".format(epoch, epochs - 1))

        running_loss = 0.0
        running_corrects = 0.0

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            preds = torch.max(outputs, 1)[1]

            loss.backward()
            # clip_grad_norm_(self.mlp.parameters(), 1)

            optimizer.step()

            # print(outputs)

            running_loss += loss.item() * inputs.size(0) / len(train_set)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss
        epoch_acc = running_corrects.double() / len(train_set)

        print("Loss: {:.6f} Acc: {:.6f}".format(epoch_loss, epoch_acc))


def train_model(
    model,
    train_set,
    epochs=30,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-3,
    step=0,
    loss_function=nn.CrossEntropyLoss,
    device="cpu",
    ignore_label=-100,
    only_classifier=False,
    wandb=None,
):

    # torch.manual_seed(42)
    # np.random.seed(42)
    # if device != 'cpu':
    #    torch.backends.cudnn.deterministic = True

    dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model.to(device)
    model.eval()

    criterion = loss_function(ignore_index=ignore_label)

    parameters = []

    if not only_classifier:
        model.feature_extractor.train()
        parameters.append(
            {
                "params": model.feature_extractor.parameters(),
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
    model.classifier.train()
    parameters.append(
        {
            "params": model.classifier.parameters(),
            "lr": lr,
            "weight_decay": weight_decay,
        }
    )

    # optimizer
    optimizer = optim.Adam(parameters)
    if step > 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.1)

    # training
    print(f"Training classifier for {epochs} epochs")
    for epoch in range(0, epochs):
        print("-" * 40)
        print("Epoch {}/{}".format(epoch, epochs - 1))

        running_loss = 0.0
        running_corrects = 0.0
        n = 0

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.max(outputs, 1)[1]

            loss.backward()

            if epoch < 3:
                nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            # print(outputs)
            mask = labels != ignore_label

            running_loss += loss.item() * (mask.sum())
            running_corrects += torch.sum(preds[mask] == labels[mask].data)

            n += (mask).sum()

        if step > 0:
            scheduler.step()

        epoch_loss = running_loss / n
        epoch_acc = (running_corrects.double()) / n

        if wandb:
            wandb.log({"loss": epoch_loss, "train-acc": epoch_acc}, step=epoch)

        print("Loss: {:.6f} Acc: {:.6f}".format(epoch_loss, epoch_acc))

        # if epoch_acc >= 0.9900000:
        #    break


def save_model(model, outputs_dir, model_filename):
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    dir_to_save = os.path.join(outputs_dir, model_filename)

    print("Saving model...")
    torch.save(model.state_dict(), dir_to_save)


def load_model(
    model_path, architecture, input_shape, remove_border=0, default_std=1e-6
):
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    creator = LCNCreator(
        architecture,
        input_shape=input_shape,
        remove_border=remove_border,
        default_std=default_std,
        relabel_markers=False,
    )
    print("Loading model...")
    creator.load_model(state_dict)

    model = creator.get_LIDSConvNet()

    return model


def load_torchvision_model_weights(model, weigths_path):
    state_dict = torch.load(weigths_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    return model


def shift_weights(in_weights, layer):

    kn = 1 #neutral element
    for i in range(len(layer.kernel_size)):
        kn = kn * layer.kernel_size[i]
    
    mid_index = int(kn/2)+1


    in_channels = layer.in_channels


    weights = np.concatenate((in_weights[in_channels:mid_index*in_channels],in_weights[0:in_channels],in_weights[mid_index*in_channels:]), axis=0)
    return weights


def unshift_weights(in_weights, layer):

    kn = 1 #neutral element
    for i in range(len(layer.kernel_size)):
        kn = kn * layer.kernel_size[i]
    
    mid_index = int(kn/2)


    in_channels = layer.in_channels

    weights = np.concatenate((in_weights[mid_index*in_channels:(mid_index+1)*in_channels],in_weights[0:mid_index*in_channels],in_weights[(mid_index+1)*in_channels:]), axis=0)
    return weights

def load_weights_from_lids_model(model, lids_model_dir):
    print("Loading LIDS model...")

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Sequential, ParallelModule)):
            continue
        print(name)
        name = name.split(".")[-1]
        if isinstance(layer, MarkerBasedNorm2d):
            conv_name = name.replace("m-norm", "conv")
            with open(os.path.join(lids_model_dir, f"{conv_name}-mean.txt")) as f:
                lines = f.readlines()[0]
                mean = np.array(
                    [float(line) for line in lines.split(" ") if len(line) > 0]
                )

            with open(os.path.join(lids_model_dir, f"{conv_name}-stdev.txt")) as f:
                lines = f.readlines()[0]
                std = np.array(
                    [float(line) for line in lines.split(" ") if len(line) > 0]
                )

            layer.mean_by_channel = torch.from_numpy(mean).float()
            layer.std_by_channel = torch.from_numpy(std).float()

        if isinstance(layer, nn.Conv2d):
            if os.path.exists(os.path.join(lids_model_dir, f"{name}-kernels.npy")):
                weights = shift_weights(np.load(os.path.join(lids_model_dir, f"{name}-kernels.npy")), layer)

                in_channels = layer.in_channels
                out_channels = layer.out_channels
                kernel_size = layer.kernel_size

                weights = weights.transpose()
                weights = weights.reshape(
                    out_channels, kernel_size[1], kernel_size[0],in_channels,
                )
                weights = weights.transpose(0,3,1,2)

                layer.weight = nn.Parameter(torch.from_numpy(weights).float())

        if isinstance(layer, nn.Conv3d):

            if os.path.exists(os.path.join(lids_model_dir, f"{name}-kernels.npy")):
                weights = shift_weights(np.load(os.path.join(lids_model_dir, f"{name}-kernels.npy")), layer)

                in_channels = layer.in_channels
                out_channels = layer.out_channels
                kernel_size = layer.kernel_size

                weights = weights.transpose()
                weights = weights.reshape(
                    out_channels,
                    kernel_size[2],
                    kernel_size[1],
                    kernel_size[0],
                    in_channels
                ).transpose(0,4,1,2,3)

                layer.weight = nn.Parameter(torch.from_numpy(weights).float())

        if isinstance(layer, MarkerBasedNorm3d):
            conv_name = name.replace("m-norm", "conv")
            with open(os.path.join(lids_model_dir, f"{conv_name}-mean.txt")) as f:
                lines = f.readlines()[0]
                mean = np.array(
                    [float(line) for line in lines.split(" ") if len(line) > 0]
                )
            with open(os.path.join(lids_model_dir, f"{conv_name}-stdev.txt")) as f:
                lines = f.readlines()[0]
                std = np.array(
                    [float(line) for line in lines.split(" ") if len(line) > 0]
                )

            layer.mean_by_channel = nn.Parameter(
                torch.from_numpy(mean.reshape(1, -1, 1, 1, 1)).float()
            )
            layer.std_by_channel = nn.Parameter(
                torch.from_numpy(std.reshape(1, -1, 1, 1, 1)).float()
            )

    print("Finish loading...")
    return model


def save_lids_model(model, architecture, outputs_dir, model_name):
    if not isinstance(model, LIDSConvNet):
        pass

    print("Saving model in LIDS format...")

    if model_name.endswith(".pt"):
        model_name = model_name.replace(".pt", "")

    if not os.path.exists(os.path.join(outputs_dir, model_name)):
        os.makedirs(os.path.join(outputs_dir, model_name))

    layer_specs = get_arch_in_lids_format(architecture)
    conv_count = 1
    for _, layer in model.named_children():
        if isinstance(layer, nn.Sequential):
            norm_index = 0
            conv_index = 0
            for i in range(len(layer)):
                if (isinstance(layer[i], (MarkerBasedNorm3d,MarkerBasedNorm2d))):
                    norm_index = i
                    conv_index = i+1
                else:
                    continue

                weights = layer[conv_index].weight.detach().cpu().numpy()

                if (isinstance(layer[conv_index], Conv3d)):

                    weights = weights.transpose(0,2,3,4,1)

                else:
                    #2d
                    weights = weights.transpose(0,2,3,1)


                out_channels = weights.shape[0]
                weights = weights.reshape(out_channels, -1)

                weights = weights.transpose()

                weights = unshift_weights(weights, layer[conv_index])

                mean = layer[norm_index].mean_by_channel.detach().cpu()
                std = layer[norm_index].std_by_channel.detach().cpu()

                mean = mean.reshape(1, -1)
                std = std.reshape(1, -1)

                np.save(
                    os.path.join(outputs_dir, model_name, f"conv{conv_count}-kernels.npy"),
                    weights,
                )
                np.savetxt(
                    os.path.join(outputs_dir, model_name, f"conv{conv_count}-mean.txt"),
                    mean.float(),
                )
                np.savetxt(
                    os.path.join(outputs_dir, model_name, f"conv{conv_count}-stdev.txt"),
                    std.float(),
                )

                conv_count += 1

    for i, layer_spec in enumerate(layer_specs, 1):

        with open(
            os.path.join(outputs_dir, model_name, f"convlayerseeds-layer{i}.json"), "w"
        ) as f:
            json.dump(layer_spec, f, indent=4)

    """for name, layer in model.classifier.named_children():
        if isinstance(layer, SpecialLinearLayer):
            weights = layer._linear.weight.detach().cpu()
            
            weights.transpose(0, 1)
            
            mean = layer.mean.detach().cpu()
            std = layer.std.detach().cpu()
            
            mean = mean.reshape(-1)
            std = std.reshape(-1)
            
            np.save(os.path.join(outputs_dir, model_name, f"{name}-weights.npy"), weights.float())
            np.savetxt(os.path.join(outputs_dir, model_name, f"{name}-mean.txt"), mean.float())
            np.savetxt(os.path.join(outputs_dir, model_name, f"{name}-std.txt"), std.float())"""


def _calulate_metrics(true_labels, pred_labels):
    average = "binary" if np.unique(true_labels).shape[0] == 2 else "weighted"
    acc = 1.0 * (true_labels == pred_labels).sum() / true_labels.shape[0]
    precision, recall, f_score, support = precision_recall_fscore_support(
        true_labels, pred_labels, zero_division=0
    )
    precision_w, recall_w, f_score_w, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average=average, zero_division=0
    )
    cm = confusion_matrix(true_labels, pred_labels)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    kappa = cohen_kappa_score(true_labels, pred_labels)

    print("#" * 50)
    print(
        colored("Acc", "yellow"), f': {colored(f"{acc:.6f}", "blue", attrs=["bold"])}'
    )
    print("-" * 50)
    print(
        colored("F1-score", "yellow"),
        f': {colored(f"{f1_score(true_labels, pred_labels, average=average):.6f}", "blue", attrs=["bold"])}',
    )
    print("-" * 50)
    print("Accuracy", *cm.diagonal())
    print("-" * 50)
    print("Precision:", *precision)
    print("Recall:", *recall)
    print("F-score:", *f_score)
    print("-" * 50)
    print("W-Precision:", precision_w)
    print("W-Recall:", recall_w)
    print("W-F-score:", f_score_w)
    print("-" * 50)
    print("Kappa {}".format(kappa))
    print("-" * 50)
    print("Suport", *support)
    print("#" * 50)

    return acc, precision, recall, f_score, precision_w, recall_w, f_score_w, kappa


def validate_model(
    model, val_set, criterion=nn.CrossEntropyLoss(), batch_size=32, device="cpu"
):

    dataloader = DataLoader(
        val_set, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model.eval()
    model.to(device)

    running_loss = 0.0
    running_corrects = 0.0

    true_labels = torch.Tensor([]).long()
    pred_labels = torch.Tensor([]).long()

    print("Validating...")

    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            preds = torch.max(outputs, 1)[1]

        running_loss += loss.item() * inputs.size(0) / len(val_set)
        running_corrects += torch.sum(preds == labels.data)

        true_labels = torch.cat((true_labels, labels.cpu()))
        pred_labels = torch.cat((pred_labels, preds.cpu()))

    print("Val - loss: {:.6f}".format(running_loss))
    print("Calculating metrics...")
    _calulate_metrics(true_labels, pred_labels)


def train_svm(
    model, train_set, batch_size=32, max_iter=10000, device="cpu", C=100, degree=3
):
    print("Preparing to train SVM")
    clf = svm.SVC(
        max_iter=max_iter,
        C=C,
        degree=degree,
        gamma="auto",
        coef0=0,
        decision_function_shape="ovo",
        kernel="linear",
    )
    dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False, drop_last=False
    )

    model.eval()
    model.to(device)

    features = torch.Tensor([])
    y = torch.Tensor([]).long()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).detach()
        features = torch.cat((features, outputs.cpu()))
        y = torch.cat((y, labels.cpu().long()))

    print("Fitting SVM...")
    clf.fit(features.flatten(start_dim=1), y)

    print("Done")
    return clf


def save_svm(clf, outputs_dir, svm_filename):
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    dir_to_save = os.path.join(outputs_dir, svm_filename)
    print("Saving SVM...")
    joblib.dump(clf, dir_to_save, compress=9)


def load_svm(svm_path):
    print("Loading SVM...")
    clf = joblib.load(svm_path)

    return clf


def validate_svm(model, clf, val_set, batch_size=32, device="cpu"):
    dataloader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False
    )

    model.eval()
    model.to(device)

    true_labels = torch.Tensor([]).long()
    pred_labels = torch.Tensor([]).long()

    for i, data in enumerate(dataloader, 0):
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        if hasattr(model, "features"):
            outputs = model.features(inputs).detach()
        else:
            outputs = model(inputs).detach()

        preds = clf.predict(outputs.cpu().flatten(start_dim=1))

        true_labels = torch.cat((true_labels, labels.cpu()))
        pred_labels = torch.cat((pred_labels, torch.from_numpy(preds)))

    print("Calculating metrics...")
    _calulate_metrics(true_labels, pred_labels)


def _images_close_to_center(images, centers):
    _images = []
    for center in centers:
        _center = np.expand_dims(center, 0)

        dist = cdist(images, _center)

        _images.append(images[np.argmin(dist)])

    return np.array(_images)


def _find_elems_in_array(a, elems):
    indices = []
    for elem in elems:
        _elem = np.expand_dims(elem, 0)
        mask = np.all(a == _elem, axis=1)

        indice = np.where(mask)[0][0:1].item()

        indices.append(indice)

    return indices


def select_images_to_put_markers(dataset, class_proportion=4, random=True):
    print("Selecting images to put markers...")
    if random:
        _selected_images = []
        images_names = np.array(dataset.images_names)
        labels = np.array([int(name.split("_")[0]) for name in images_names])

        for label in np.unique(labels):
            images_of_label = np.argwhere(labels == label).squeeze()
            if isinstance(class_proportion, int):
                n = class_proportion
            else:
                n = images_of_label.shape[0] * class_proportion

            selected_of_label = np.random.choice(images_of_label, n, replace=False)

            _selected_images.extend(selected_of_label)

        selected_images = np.array([dataset[i][0] for i in _selected_images])
        images_names = [dataset.images_names[i] for i in _selected_images]

    else:
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)

        all_images = None
        all_labels = None

        input_shape = dataset[0][0].shape

        for images, labels in dataloader:
            if all_images is None:
                all_images = images
                all_labels = labels
            else:
                all_images = torch.cat((all_images, images))
                all_labels = torch.cat((all_labels, labels))

        all_images = all_images.flatten(1).numpy()
        all_labels = all_labels.numpy()

        possible_labels = np.unique(all_labels)

        images_names = []

        roots = None

        for label in possible_labels:
            images_of_label = all_images[all_labels == label]
            if isinstance(class_proportion, float):
                n_clusters = max(
                    1, math.floor(images_of_label.shape[0] * class_proportion)
                )
            else:
                n_clusters = class_proportion
            print(n_clusters)

            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(images_of_label)

            roots_of_label = _images_close_to_center(
                images_of_label, kmeans.cluster_centers_
            )

            if roots is None:
                roots = roots_of_label
            else:
                roots = np.concatenate((roots, roots_of_label))

            indices = _find_elems_in_array(all_images, roots_of_label)

            for indice in indices:
                images_names.append(dataset.images_names[indice])

            selected_images = roots.reshape(-1, *input_shape)

    return selected_images, images_names


def _label_of_image(image_name):
    if not isinstance(image_name, str):
        raise TypeError("Parameter image_name must be a string.")
    i = image_name.index("_")
    label = int(image_name[0:i]) - 1

    return label


def split_dataset(dataset_dir, train_size, val_size=0, test_size=None, stratify=True):
    if os.path.exists(os.path.join(dataset_dir, "files.txt")):
        with open(os.path.join(dataset_dir, "files.txt"), "r") as f:
            filenames = f.read().split("\n")
        filenames = [filename for filename in filenames if len(filename) > 0]
    else:
        filenames = os.listdir(dataset_dir)
        filenames.sort()
    labels = np.array([_label_of_image(filename) for filename in filenames])

    if train_size > 1:
        train_size = int(train_size)

    train_split, test_split, _, test_labels = train_test_split(
        filenames, labels, train_size=train_size, test_size=test_size, stratify=labels
    )

    val_size = 0 if val_size is None else val_size

    val_split = []

    if val_size > 0:
        test_size = len(test_split) - val_size

        test_size = int(test_size) if test_size > 0 else test_size

        val_split, test_split = train_test_split(
            test_split, test_size=test_size, stratify=test_labels
        )

    return train_split, val_split, test_split


def compute_grad_cam(model, image, target_layers, class_label=0, device="cpu"):
    model = model.to(device)
    image = image.to(device)

    model.eval()

    gradients = []
    features = []

    if image.dim() == 3:
        x = image.unsqueeze(0)
    else:
        x = image

    for name, module in model._modules.items():
        if name == "features" or name == "feature_extractor":
            for layer_name, layer in module.named_children():
                x = layer(x)
                if layer_name in target_layers:
                    x.register_hook(lambda grad: gradients.append(grad))
                    features.append(x)

        elif name == "classifier":
            x = x.flatten(1)
            x = module(x)

        else:
            x = module(x)
    y = x

    one_hot = torch.zeros_like(y, device=device)
    one_hot[0][class_label] = 1
    one_hot = torch.sum(one_hot * y)

    model.zero_grad()
    one_hot.backward()

    weights = torch.mean(gradients[-1], axis=(2, 3))[0, :]
    target = features[-1][0].detach()

    cam = torch.zeros_like(target[0])

    for i, w in enumerate(weights):
        cam += (
            w * target[i, :,]
        )

    cam[cam < 0] = 0.0
    print(cam.shape)
    print(image.shape)
    resize = Resize(image.shape[1:])

    cam = resize(cam.unsqueeze(0))
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam.cpu().numpy()


def save_opf_dataset(path, opf_dataset):
    assert ift is not None, "PyIFT is not available"

    ift.WriteDataSet(opf_dataset, path)


def load_opf_dataset(path):
    assert ift is not None, "PyIFT is not available"

    opf_dataset = ift.ReadDataSet(path)

    return opf_dataset


def save_intermediate_outputs(
    model,
    dataset,
    outputs_dir,
    batch_size=16,
    layers=None,
    only_features=True,
    format="mimg",
    remove_border=0,
    device="cpu",
):

    if only_features:
        if hasattr(model, "features"):
            _model = model.features
        else:
            _model = model.feature_extractor
    else:
        _model = model

    last_layer = None
    for layer_name in layers:
        layer_dir = os.path.join(outputs_dir, "intermediate-outputs", layer_name)

        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)

        last_layer = layer_name

    _model.eval()
    _model.to(device)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    outputs = {}
    outputs_count = {}
    outputs_names = dataset.images_names

    print("Saving intermediate outputs...")

    for inputs, _ in dataloader:

        inputs = inputs.to(device)

        for layer_name, layer in _model.named_children():
            _outputs = layer(inputs)

            if layer_name == last_layer and remove_border > 0:
                b = remove_border

                _outputs = _outputs[:, :, b:-b, b:-b]
            inputs = _outputs

            if layer_name not in outputs_count:
                outputs_count[layer_name] = 0

            if layers is None or len(layers) == 0 or layer_name in layers:
                if format == "zip":
                    if layer_name not in outputs:
                        outputs[layer_name] = _outputs.detach().cpu()
                    else:
                        outputs[layer_name] = torch.cat(
                            (outputs[layer_name], _outputs.detach().cpu())
                        )
                elif format in ["mimg", "npy"]:
                    layer_dir = os.path.join(
                        outputs_dir, "intermediate-outputs", layer_name
                    )
                    _outputs = _outputs.detach().cpu()

                    for _output in _outputs:
                        _output_dir = os.path.join(
                            layer_dir,
                            f"{outputs_names[outputs_count[layer_name]].split('.')[0]}.{format}",
                        )

                        if format == "npy":
                            np.save(_output_dir, _output)
                        else:
                            save_mimgage(_output_dir, _output.permute(1, 2, 0).numpy())

                        outputs_count[layer_name] += 1

                    del _outputs

        torch.cuda.empty_cache()

    if format == "zip":
        for layer_name in outputs:

            _outputs = outputs[layer_name]

            _outputs = (
                _outputs.permute(0, 2, 3, 1).numpy().reshape(_outputs.shape[0], -1)
            )

            labels = np.array(
                [
                    int(image_name[0 : image_name.index("_")]) - 1
                    for image_name in outputs_names
                ]
            ).astype(np.int32)

            opf_dataset = ift.CreateDataSetFromNumPy(_outputs, labels + 1)

            opf_dataset.SetNClasses = labels.max() + 1

            ift.SetStatus(opf_dataset, ift.IFT_TRAIN)
            ift.AddStatus(opf_dataset, ift.IFT_SUPERVISED)

            # opf_dataset.SetLabels(labels + 1)

            _output_dir = os.path.join(layer_dir, "dataset.zip")
            save_opf_dataset(_output_dir, opf_dataset)


def get_arch_operations(architecture,key,operations):

    if not(isinstance(architecture, dict)):
        return operations

    if (key == "classifier"): #classifier does not belong to the IFT format
        return operations

    keys = list(architecture.keys())

    if ("operation" in keys):
        operations.append((key,architecture))
        return operations

    for k in keys:
        operations = get_arch_operations(architecture[k],k,operations)

    return operations


def get_arch_in_lids_format(architecture):

    operations = get_arch_operations(architecture,"",[])

    conv_layers_count = 1
    layer_idx = "1"
    current_layer = "layer1"
    layer_spec = {}
    layer_spec[current_layer] = {}
    lids_layer_specs = []
    for i in range(len(operations)):
        if (layer_idx != operations[i][0][-1]):
            lids_layer_specs.append(layer_spec)
            layer_spec = {}
            layer_idx = operations[i][0][-1]
            current_layer = "layer"+layer_idx
            layer_spec[current_layer] = {}
            
        operation = operations[i][1]["operation"]
        params = operations[i][1]["params"]

        if ("m_norm" in operation):
            norm_specs = {} 
            if ("kernel_size" in params.keys()):
                kernel_size = params["kernel_size"]
                if isinstance(kernel_size, int):
                    if (operation == "m_norm2d"):
                        norm_specs["kernelsize"] = [kernel_size, kernel_size, 0]
                    elif (operation == "m_norm3d"):
                        norm_specs["kernelsize"] = [kernel_size, kernel_size, kernel_size]
                else:
                    if (operation == "m_norm2d"):
                        norm_specs["kernelsize"] = [*kernel_size, 0]
                    elif (operation == "m_norm3d"):
                        norm_specs["kernelsize"] = [*kernel_size]
            if ("dilation" in params.keys()):
                dilation = params["dilation"]
                if isinstance(dilation, int):
                    if (operation == "m_norm2d"):
                        norm_specs["dilationrate"] = [dilation, dilation, 0]
                    elif (operation == "m_norm3d"):
                        norm_specs["dilationrate"] = [dilation, dilation, dilation]
                else:
                    if (operation == "m_norm2d"):
                        norm_specs["dilationrate"] = [*dilation, 0]
                    elif (operation == "m_norm3d"):
                        norm_specs["dilationrate"] = [*dilation]
            if ("epsilon" in params.keys()):
                norm_specs["epsilon"] = params["epsilon"]

            layer_spec[current_layer]["m_norm"] = norm_specs

        if ("conv" in operation):
            conv_specs = {}
            kernel_size = params["kernel_size"]
            dilation = params["dilation"]
            number_of_kernels_per_markers = params.get(
                "number_of_kernels_per_marker", 8
            )
            out_channels = params["out_channels"]

            if isinstance(kernel_size, int):
                if (operation == "conv2d"):
                    conv_specs["kernelsize"] = [kernel_size, kernel_size, 0]
                elif (operation == "conv3d"):
                    conv_specs["kernelsize"] = [kernel_size, kernel_size, kernel_size]
            else:
                if (operation == "conv2d"):
                    conv_specs["kernelsize"] = [*kernel_size, 0]
                elif (operation == "conv3d"):
                    conv_specs["kernelsize"] = [*kernel_size]

            if isinstance(dilation, int):
                if (operation == "conv2d"):
                    conv_specs["dilationrate"] = [dilation, dilation, 0]
                elif (operation == "conv3d"):
                    conv_specs["dilationrate"] = [dilation, dilation, dilation]
            else:
                if (operation == "conv2d"):
                    conv_specs["dilationrate"] = [*dilation, 0]
                elif (operation == "conv3d"):
                    conv_specs["dilationrate"] = [*dilation]

            conv_specs["nkernelspermarker"] = number_of_kernels_per_markers
            conv_specs["finalnkernels"] = out_channels
            conv_specs["nkernelsperimage"] = 10000                    
            layer_spec[current_layer]["conv"] = conv_specs
            conv_layers_count += 1
            continue

        if (operation == "relu"):
            if (operation == "true"):
                layer_spec[current_layer]["relu"] = 1
            else:
                layer_spec[current_layer]["relu"] = 0
            continue

        if ("pool" in operation):
            pool_spec = {}
            if ("max" in operation):
                pool_spec["pool_type"] = 2
            elif ("avg" in operation):
                pool_spec["pool_type"] = 1

            kernel_size = params["kernel_size"]
            stride = params["stride"]

            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]

            pool_spec["poolxsize"] = kernel_size[0]
            pool_spec["poolysize"] = kernel_size[1]
            pool_spec["poolzsize"] = 0
            if ("3d" in operation):
                pool_spec["poolzsize"] = kernel_size[2]

            pool_spec["stride"] = stride

            layer_spec[current_layer]["pooling"] = pool_spec

    lids_layer_specs.append(layer_spec)


    '''
    layer_names = list(architecture["features"]["layers"].keys())

    layers = architecture["features"]["layers"]

    lids_layer_specs = []
    conv_layers_count = 1
    
    operations = [layers[layer_name]["operation"] for layer_name in layer_names]
    conv_layers_count = 1

    for i in range(len(layer_names)):
        layer_spec = {}
        if operations[i] == "conv2d":

            params = layers[layer_names[i]]["params"]
            kernel_size = params["kernel_size"]
            dilation = params["kernel_size"]
            number_of_kernels_per_markers = params.get(
                "number_of_kernels_per_marker", 8
            )
            out_channels = params["out_channels"]

            layer_spec["layer"] = conv_layers_count

            if isinstance(kernel_size, int):
                layer_spec["kernelsize"] = [kernel_size, kernel_size, 0]
            else:
                layer_spec["kernelsize"] = [*kernel_size, 0]

            if isinstance(dilation, int):
                layer_spec["dilationrate"] = [dilation, dilation, 0]
            else:
                layer_spec["dilationrate"] = [*dilation, 0]

            layer_spec["nkernelspermarker"] = number_of_kernels_per_markers
            layer_spec["finalnkernels"] = out_channels
            layer_spec["nkernelsperimage"] = 10000

            if i + 1 < len(layer_names) and operations[i + 1] == "relu":
                layer_spec["relu"] = 1
            else:
                layer_spec["relu"] = 0

            conv_layers_count += 1

            j = i + 1 if layer_spec["relu"] == 0 else i + 2

            pool_spec = {}

            if j < len(layer_names) and "pool" in operations[j]:
                if operations[j] == "max_pool2d":
                    pool_spec["pool_type"] = 2
                elif operations[j] == "avg_pool2d":
                    pool_spec["pool_type"] = 1

                pool_params = layers[layer_names[j]]["params"]

                kernel_size = pool_params["kernel_size"]
                stride = pool_params["stride"]

                if isinstance(kernel_size, int):
                    kernel_size = [kernel_size, kernel_size]

                pool_spec["poolxsize"] = kernel_size[0]
                pool_spec["poolysize"] = kernel_size[1]
                pool_spec["poolzsize"] = 0

                pool_spec["stride"] = stride
            else:
                pool_spec["pool_type"] = 0

            layer_spec["pooling"] = pool_spec

            lids_layer_specs.append(layer_spec)
    '''

    return lids_layer_specs


def create_arch(layers_dir):
    layers_info_files = [f for f in os.listdir(layers_dir) if f.endswith(".json")]
    layers_info_files.sort()

    arch = OrderedDict([("features", {"type": "sequential", "layers": OrderedDict()})])

    layers = arch["features"]["layers"]

    for i, layer_info_file in enumerate(layers_info_files, 1):
        with open(os.path.join(layers_dir, layer_info_file), "r") as f:
            layer_info = json.load(f)

        # print(layer_info)

        conv_spec = {
            "operation": "conv2d",
            "params": {
                "kernel_size": layer_info["kernelsize"][:-1],
                "number_of_kernels_per_marker": layer_info["nkernelspermarker"],
                "dilation": layer_info["dilationrate"][:-1],
                "out_channels": layer_info["finalnkernels"],
                "padding": [
                    floor(
                        (
                            layer_info["kernelsize"][0]
                            + (layer_info["kernelsize"][0] - 1)
                            * (layer_info["dilationrate"][0] - 1)
                        )
                        / 2
                    ),
                    floor(
                        (
                            layer_info["kernelsize"][1]
                            + (layer_info["kernelsize"][1] - 1)
                            * (layer_info["dilationrate"][1] - 1)
                        )
                        / 2
                    ),
                ],
                "stride": 1,
            },
        }

        if layer_info["relu"] == 1:
            relu_spec = {"operation": "relu", "params": {"inplace": True}}
        else:
            relu_spec = None

        if layer_info["pooling"]["pooltype"] != 0:
            pool_spec = {
                "params": {
                    "kernel_size": [
                        layer_info["pooling"]["poolxsize"],
                        layer_info["pooling"]["poolysize"],
                    ],
                    "stride": layer_info["pooling"]["poolstride"],
                    "padding": [
                        floor(layer_info["pooling"]["poolxsize"] / 2),
                        floor(layer_info["pooling"]["poolysize"] / 2),
                    ],
                }
            }

            if layer_info["pooling"]["pooltype"] == 2:
                pool_spec["operation"] = "max_pool2d"

            elif layer_info["pooling"]["pooltype"] == 1:
                pool_spec["operation"] = "avg_pool2d"

        layers[f"conv{i}"] = conv_spec

        if relu_spec is not None:
            layers[f"relu{i}"] = relu_spec

        if pool_spec is not None:
            layers[f"pool{i}"] = pool_spec

    return arch


def save_arch(arch, output_path):
    dirname = os.path.dirname(output_path)

    if not os.path.exists(dirname) and dirname != "":
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, "w") as f:
        json.dump(arch, f, indent=4)
