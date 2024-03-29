import warnings

import nibabel as nib
import numpy as np
from PIL import Image
from skimage.color import gray2rgb, lab2rgb, rgba2rgb
from skimage.util import img_as_float

try:
    import pyift.pyift as ift
except ModuleNotFoundError:
    ift = None
    warnings.warn("pyift is not installed.", RuntimeWarning)

__all__ = [
    "load_image",
    "image_to_rgb",
    "from_lab_to_rgb",
    "load_mimage",
    "save_mimage",
    "image_to_lab",
]


def _labf(x):
    if x >= 8.85645167903563082e-3:
        return x ** (0.33333333333)
    else:
        return (841.0 / 108.0) * (x) + (4.0 / 29.0)


def image_to_lab(image):
    return _image_to_lab(image)


def _image_to_lab(image):
    image = image / image.max()

    labf_v = np.vectorize(_labf)

    new_image = np.zeros_like(image)

    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    X = (
        0.4123955889674142161 * R
        + 0.3575834307637148171 * G
        + 0.1804926473817015735 * B
    )
    Y = (
        0.2125862307855955516 * R
        + 0.7151703037034108499 * G
        + 0.07220049864333622685 * B
    )
    Z = (
        0.01929721549174694484 * R
        + 0.1191838645808485318 * G
        + 0.9504971251315797660 * B
    )

    X = labf_v(X / 0.950456)
    Y = labf_v(Y / 1.0)
    Z = labf_v(Z / 1.088754)

    new_image[:, :, 0] = 116 * Y - 16
    new_image[:, :, 1] = 500 * (X - Y)
    new_image[:, :, 2] = 200 * (Y - Z)

    # new_image = rgb2lab(image)

    new_image[:, :, 0] = new_image[:, :, 0] / 99.998337
    new_image[:, :, 1] = (new_image[:, :, 1] + 86.182236) / (86.182236 + 98.258614)
    new_image[:, :, 2] = (new_image[:, :, 2] + 107.867744) / (107.867744 + 94.481682)

    return new_image


def load_and_convert_2d_images(path, lab: bool = True) -> np.ndarray:
    image = Image.open(path)
    image = np.array(image)
    if lab:
        if image.ndim == 3 and image.shape[-1] == 4:
            image = rgba2rgb(image)
        elif image.ndim == 2 or image.shape[-1] == 1:
            image = gray2rgb(image)
        elif image.ndim == 3 and image.shape[-1] > 4:
            image = gray2rgb(image)
        elif image.ndim == 4 and image.shape[-1] == 4:
            image = rgba2rgb(image)

        image = _image_to_lab(image)

        if image.dtype != float:
            image = img_as_float(image)

    return image


def load_image(path: str, lab: bool = True) -> np.ndarray:
    if path.endswith(".mimg"):
        image = load_mimage(path)
    elif ift is not None:
        image = ift.ReadImageByExt(path)
        if ift.Is3DImage(image):
            if lab:
                mimage = ift.ImageToMImage(image, color_space=ift.LABNorm2_CSPACE)
                image = mimage.AsNumPy()
            else:
                image = image.AsNumPy()
            image = image.squeeze()
        else:  # 2D Image
            # Apparently, load_and_convert_2d_images and
            # the code below should do the same, but this is not
            # happening.

            image = load_and_convert_2d_images(path, lab)

    #           if lab:
    #              mimage = ift.ImageToMImage(image, \
    #                                          color_space=ift.LABNorm2_CSPACE)
    #              image = mimage.AsNumPy().transpose(0,1,2,3)
    #           else:
    #              image = image.AsNumPy()
    #           image = image.squeeze()

    elif path.endswith(".nii.gz"):
        if lab:
            warnings.warn(
                "pyift is required to convert 3D images into lab", UserWarning
            )
        image = nib.load(path)
        image = image.get_fdata().transpose(1, 0, 2)
    else:  # For 2D images when pyift is not available
        image = load_and_convert_2d_images(path, lab)

    return image


def image_to_rgb(image):
    warnings.warn(
        "'image_to_rgb' will be remove due to its misleading name. "
        + "Use 'from_lab_to_rgb' instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return from_lab_to_rgb(image)


def from_lab_to_rgb(image):
    image = lab2rgb(image)
    return image


def load_mimage(path):
    assert ift is not None, "PyIFT is not available"

    mimage = ift.ReadMImage(path)

    return mimage.AsNumPy().squeeze()


def save_mimage(path, image):
    assert ift is not None, "PyIFT is not available"

    mimage = ift.CreateMImageFromNumPy(np.ascontiguousarray(image))
    ift.WriteMImage(mimage, path)
