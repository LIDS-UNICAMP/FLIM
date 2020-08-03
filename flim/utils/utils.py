import numpy as np
import torch

from skimage.util import view_as_windows, pad
from skimage.color import rgb2lab, lab2rgb
from skimage import io

from sklearn.metrics import precision_recall_fscore_support, jaccard_score

from scipy.ndimage import label

import math

import json

import os


def label_connected_componentes(label_image):
    label_image = label_image.squeeze()

    num_labels = label_image.astype(np.int32).max()

    new_label_image = np.zeros_like(label_image)
    structure = np.ones((3, 3), dtype=np.uint8)

    c = 1

    for l in range(1, num_labels + 1):
        _label_image = label(label_image == l, structure=structure)[0]

        for new_l in range(1, _label_image.max() + 1):
            mask = _label_image == new_l
            new_label_image[mask] = c
            c += 1

    return np.expand_dims(new_label_image, 0)
        
