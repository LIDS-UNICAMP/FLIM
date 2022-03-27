import warnings
import os
from ._image_utils import load_image

from torch.utils.data import Dataset

import torch

import numpy as np

from skimage.color import rgb2lab

try:
    import pyift.pyift as ift
except:
    warnings.warn("PyIFT is not installed.", ImportWarning)

__all__ = ["LIDSDataset", "ToTensor", "ToLAB"]


class LIDSDataset(Dataset):
    def __init__(
        self, root_dir, split_dir=None, lab=True, transform=None, return_name=False
    ):
        self.root_dir = root_dir
        self.split_dir = split_dir
        self.transform = transform

        self.return_name = return_name
        self.images_names = None
        self.opf_data = None
        self.opf_labels = None

        self._lab = lab

        if self.root_dir.endswith(".zip"):
            self.opf_data, self.opf_labels = self._get_data_from_opfdataset()
        else:
            self.images_names = self._list_images_files()

    def __len__(self):
        if self.opf_data is not None:
            return self.opf_data.shape[0]
        return len(self.images_names)

    def __getitem__(self, idx):
        if self.opf_data is None:
            image_path = os.path.join(self.root_dir, self.images_names[idx])

            image = load_image(image_path, lab=self._lab)

            label = self._label_of_image(self.images_names[idx])

        else:

            image = self.opf_data[idx]
            label = self.opf_labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.return_name:
            sample = (image, label, self.images_names[idx])
        else:
            sample = (image, label)

        return sample

    def _label_of_image(self, image_name):
        if not isinstance(image_name, str):
            raise TypeError("Parameter image_name must be a string.")
        i = image_name.index("_")
        label = int(image_name[0:i]) - 1

        return label

    def _list_images_files(self):
        if self.split_dir is not None:
            with open(self.split_dir, "r") as f:
                _filenames = f.read()
                filenames = [
                    filename for filename in _filenames.split("\n") if len(filename) > 0
                ]
        else:

            filenames = os.listdir(self.root_dir)

        return filenames

    def _get_data_from_opfdataset(self):
        opfdataset = ift.ReadDataSet(self.root_dir)

        data = opfdataset.GetData()
        labels = opfdataset.GetTrueLabels().astype(np.int64) - 1

        return data, labels

    def weights_for_balance(self, nclasses):
        weights_dir = os.path.join(
            self.root_dir, ".weights-for-balance-{}.npy".format(self.image_list)
        )

        if os.path.exists(weights_dir):
            weight = np.load(weights_dir)
            return weight

        count = [0] * nclasses
        for image_name in self.images_names:
            label = self._label_of_image(image_name)
            count[label] += 1
        weight_per_class = [0.0] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * self.__len__()
        for idx, image_name in enumerate(self.images_names):
            label = self._label_of_image(image_name)
            weight[idx] = weight_per_class[label]
        weight = np.array(weight, dtype=np.float)

        np.save(weights_dir, weight)

        return weight


class ToTensor(object):
    def __call__(self, sample):
        image = np.array(sample)

        # min = image.min()
        # max = image.max()
        # image = (image) / (max)

        if image.ndim > 2:
            image = image.transpose((2, 0, 1))

        return torch.from_numpy(image.copy()).float()


class ToLAB(object):
    def __call__(self, sample):
        image = np.array(sample)

        image = rgb2lab(image)
        return image
