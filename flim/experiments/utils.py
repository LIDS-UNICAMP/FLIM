import json
from logging import root

import os

from skimage import io
from skimage.color import rgb2lab

import numpy as np

from numba import njit

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from sklearn.metrics import f1_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans

from scipy.spatial.distance import cdist

import joblib

from termcolor import colored

import math

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

def build_model(architecture,
                images,
                markers,
                batch_size=32,
                train_set=None,
                device='cpu'):
    creator = LCNCreator(architecture,
                         images=images,
                         markers=markers,
                         batch_size=batch_size,
                         relabel_markers=False,
                         device=device)

    print("Building feature extractor...")
    creator.build_feature_extractor()

    if "classifier" in architecture:
        print("Building classifier...")
        creator.build_classifier(train_set)

    model = creator.get_LIDSConvNet()

    print("Model ready.")

    return model


def train_mlp(model,
              train_set,
              epochs=30,
              batch_size=64,
              lr=1e-3,
              weight_decay=1e-3,
              criterion=nn.CrossEntropyLoss(),
              device='cpu'):


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


def save_model(model, outputs_dir, model_filename):
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    dir_to_save = os.path.join(outputs_dir, model_filename)

    print("Saving model...")
    torch.save(model.state_dict(), dir_to_save)

def load_model(model_path, architecture, input_shape):
    state_dict = torch.load(model_path)

    creator = LCNCreator(architecture,
                         input_shape=input_shape,
                         relabel_markers=False)
    print("Loading model...")
    creator.load_model(state_dict)

    model = creator.get_LIDSConvNet()

    return model

def _calulate_metrics(true_labels, pred_labels):
    acc = 1.0*(true_labels == pred_labels).sum()/true_labels.shape[0]
    precision, recall, f_score, support = precision_recall_fscore_support(true_labels, pred_labels, zero_division=0)
    precision_w, recall_w, f_score_w, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', zero_division=0)

    print("#" * 50)
    print(colored("Acc", "yellow"),f': {colored(f"{acc:.6f}", "blue", attrs=["bold"])}')
    print("-" * 50)
    print(colored("F1-score", "yellow"), f': {colored(f"{f1_score(true_labels, pred_labels):.6f}", "blue", attrs=["bold"])}')
    print("-" * 50)
    print("Precision:", *precision)
    print("Recall:", *recall)
    print("F-score:", *f_score)
    print("-" * 50)
    print("W-Precision:", precision_w)
    print("W-Recall:", recall_w)
    print("W-F-score:", f_score_w)
    print("-" * 50)
    print("Kappa {}".format(cohen_kappa_score(true_labels, pred_labels)))
    print("-" * 50)
    print("Suport", *support)
    print("#" * 50)

def validate_model(model,
                   val_set,
                   criterion=nn.CrossEntropyLoss(),
                   batch_size=32,
                   device='cpu'):

    dataloader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=True)

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
            

        running_loss += loss.item()*inputs.size(0)/len(val_set)
        running_corrects += torch.sum(preds == labels.data)
        
        true_labels = torch.cat((true_labels, labels.cpu()))
        pred_labels = torch.cat((pred_labels, preds.cpu()))
    
    print('Val - loss: {:.6f}'.format(running_loss))
    print("Calculating metrics...")
    _calulate_metrics(true_labels, pred_labels)

def train_svm(model, train_set, batch_size=32, device='cpu'):
    clf = svm.LinearSVC(max_iter=50000)
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    features = torch.Tensor([])
    y = torch.Tensor([]).long()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).detach()
        features = torch.cat((features, outputs.cpu()))
        y = torch.cat((y, labels.cpu()))
    
    print("Fitting SVM...")
    clf.fit(features.flatten(start_dim=1), y)

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

def validate_svm(model, clf, val_set, batch_size=32, device='cpu'):
    dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    true_labels = torch.Tensor([]).long()
    pred_labels = torch.Tensor([]).long()

    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        
        inputs, labels = inputs.to(device), labels.to(device)
        
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


def select_images_to_put_markers(dataset, class_proportion=0.05):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

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
        n_clusters = max(1, math.floor(images_of_label.shape[0]*class_proportion))
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(images_of_label)

        roots_of_label = _images_close_to_center(images_of_label, kmeans.cluster_centers_)

        if roots is None:
            roots = roots_of_label
        else:
            roots = np.concatenate((roots, roots_of_label))
        
        indices = _find_elems_in_array(all_images, roots_of_label)

        for indice in indices:
            images_names.append(dataset.images_names[indice])


    return roots.reshape(-1, *input_shape), images_names

