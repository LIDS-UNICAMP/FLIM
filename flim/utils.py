"""Helper functions.

Helper functions to suport other modules.

"""
import numpy as np

from scipy.ndimage import label

__all__ = ['label_connected_componentes']

def label_connected_componentes(label_image):
    """Label connected components in a label image

    Create a new label image where labels are changed so that each connected component has a diffetent label.
    To find the connected components, it is used a 8-neighborhood.

    Parameters
    ----------
    label_image : ndarray
        Label image with size :math:`(H, W)`.

    Returns
    -------
    ndarray
        A label image with new labels.
    """    
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
        
