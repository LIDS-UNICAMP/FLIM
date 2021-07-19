"""Helper functions.

Helper functions to suport other modules.

"""
import numpy as np

from scipy.ndimage import label

import torch

__all__ = ['label_connected_components', 'compute_importance']


def label_connected_components(label_images, start_label=1, is_3d=False):
    """Label connected components in a label image.

    Create new label images where labels are changed so that each \
    connected componnent has a diffetent label. \
    To find the connected components, it is used a 8-neighborhood.

    Parameters
    ----------
    label_images : ndarray
        Label images with size :math:`(N, H, W)`.

    start_labeel: int
        First label, by default 1.

    Returns
    -------
    ndarray
        Label images with new labels.

    """
    # label_image = label_image.squeeze()

    if label_images.ndim == 2:
        label_images = np.expand_dims(label_images, 0)

    if is_3d and label_images.ndim == 3:
        label_images = np.expand_dims(label_images, 0)

    new_label_images = np.zeros_like(label_images).astype(np.int)

    _c = start_label

    for label_image, new_label_image in zip(label_images, new_label_images):
        num_labels = label_image.astype(np.int32).max()

        #new_label_image = np.zeros_like(label_image).astype(np.int)
        if is_3d:
            structure = np.ones((3, 3, 3), dtype=np.uint8)
        else:
            structure = np.ones((3, 3), dtype=np.uint8)

        for _l in range(1, num_labels + 1):
            _label_image = label(label_image == _l, structure=structure)[0]

            for new_l in range(1, _label_image.max() + 1):
                mask = _label_image == new_l
                new_label_image[mask] = _c
                _c += 1

    return new_label_images

def _normalize_image(image):
    f = image
    
    max = f.max()
    min = f.min()
    if max - min > 0:
        f = image
        f = 1*(f - min)/(max - min)
    elif max > 0:
        f = f/max

    return f

def _compute_channel_importance(image_features, image_markers, label):
    importance_to_label = {}
    num_channels = image_features[0].shape[0]

    pos_act_by_channel = np.zeros(num_channels)
    neg_act_by_channel = np.zeros(num_channels)

    importance_to_label = np.zeros(num_channels)
    num_label_pixels = 0
    num_other_label_pixels = 0

    for features, markers in zip(image_features, image_markers):
        other_label_mask = np.logical_and(markers != label, markers != 0)
        label_mask = markers == label
        num_label_pixels += label_mask.sum()
        num_other_label_pixels += other_label_mask.sum()

        for i, channel in enumerate(features):
            _channel = _normalize_image(channel)
            pos_act_by_channel[i] = (_channel[label_mask].sum())
            neg_act_by_channel[i] = 1 - _channel[other_label_mask].sum()

    # reg_factor_pos = 1 - num_label_pixels/(num_label_pixels + num_other_label_pixels)
    # reg_factor_neg = 1 - reg_factor_pos

    importance_to_label = pos_act_by_channel + neg_act_by_channel
    importance_to_label = importance_to_label - importance_to_label.mean()
    
    #importance_to_label = importance_to_label/np.linalg.norm(importance_to_label)

    return importance_to_label


def compute_importance(images, markers, n_classes):
    
    importance_by_channel = []

    images = images.transpose(0, 3, 1, 2)

    for label in range(n_classes):

        importance_by_channel.append(_compute_channel_importance(images, markers, label+1))

    return np.array(importance_by_channel)
