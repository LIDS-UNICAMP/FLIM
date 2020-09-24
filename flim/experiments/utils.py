import json

import os

from skimage import io
from skimage.color import rgb2lab

import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from ..models.lcn import LCNCreator
from ._dataset import LIDSDataset

def load_image(image_dir):
    image = rgb2lab(io.imread(image_dir))
    image = image/(np.array([[116], [500], [200]])).reshape(1, 1, 3)
    return image


def load_markers(markers_dir):
    markers = []
    lines = None
    with open(markers_dir, 'r') as f:
        lines = f.readlines()
    
    label_infos = [int(info) for info in lines[0].split(" ")]

    image_shape = (label_infos[2], label_infos[1])
    
    markers = np.zeros(image_shape, dtype=np.int)

    for line in lines[1:]:
        split_line = line.split(" ")
        x, y, label = int(split_line[0]), int(split_line[1]), int(split_line[3])

        markers[x][y] = label

    return markers

def load_images_and_markers(path):
    dirs = os.listdir(path)
    images_names = [filename for filename in dirs if not filename.endswith('.txt')]
    makers_names = [filename for filename in dirs if filename.endswith('.txt')]

    images_names.sort()
    makers_names.sort()
    
    images = []
    images_markers = []

    for image_name, marker_name in zip(images_names, makers_names):
        
        if image_name.endswith('.npy'):
            image = np.load(os.path.join(path, image_name))
        else:
            image = load_image(os.path.join(path, image_name))
        markers = load_markers(os.path.join(path, marker_name))
        
        images.append(image)
        images_markers.append(markers)

    return np.array(images), np.array(images_markers)

def load_architecture(architecture_dir):
    path = architecture_dir
    with open(path) as json_file:
        architecture = json.load(json_file)

    return architecture

def configure_dataset(dataset_dir, split_dir, transform=None):
    dataset = LIDSDataset(dataset_dir, split_dir, transform)

    return dataset

def build_model(architecture, images, markers, batch_size=32, device='cpu'):
    creator = LCNCreator(architecture,
                         images=images,
                         markers=markers,
                         batch_size=batch_size,
                         relabel_markers=False,
                         device=device)

    creator.build_feature_extractor()

    creator.build_classifier()

    model = creator.get_LIDSConvNet()

    return model


def train_mlp(model,
              train_set,
              epochs=30,
              batch_size=64,
              lr=1e-3,
              weight_decay=1e-3,
              criterion=nn.CrossEntropyLoss(),
              device='cpu',
              outputs_dir=None):


    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.to(device)
    model.feature_extractor.eval()
    model.classifier.train()

    #optimizer
    optimizer = optim.Adam(model.classifier.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)
  
    #training
    print(f"Training classifier for {epochs} epochs")
    for epoch in range(0, epochs):
        print('-' * 40)
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        
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
            #clip_grad_norm_(self.mlp.parameters(), 1)
            
            optimizer.step()
            
            #print(outputs)
            
            running_loss += loss.item()*inputs.size(0)/len(train_set)
            running_corrects += torch.sum(preds == labels.data) 
        
        #scheduler.step()
        epoch_loss = running_loss
        epoch_acc = running_corrects.double()/len(train_set)

        print('Loss: {:.6f} Acc: {:.6f}'.format(epoch_loss, epoch_acc))