"""Helper functions.

Helper functions to suport other modules.

"""
import numpy as np

from scipy.ndimage import label

__all__ = ['label_connected_components']


def label_connected_components(label_images, start_label=1):
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

    new_label_images = np.zeros_like(label_images).astype(np.int)

    for label_image, new_label_image in zip(label_images, new_label_images):
        num_labels = label_image.astype(np.int32).max()

        #new_label_image = np.zeros_like(label_image).astype(np.int)
        structure = np.ones((3, 3), dtype=np.uint8)

        _c = start_label

        for _l in range(1, num_labels + 1):
            _label_image = label(label_image == _l, structure=structure)[0]

            for new_l in range(1, _label_image.max() + 1):
                mask = _label_image == new_l
                new_label_image[mask] = _c
                _c += 1

    return new_label_images
