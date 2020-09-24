
import os

from torch.utils.data import Dataset

import numpy as np

from skimage import io
from skimage.color import rgb2lab

__all__ = ["LIDSDataset"]

class LIDSDataset(Dataset):
    def __init__(self, root_dir, split_dir, transform=None):
        self.root_dir = root_dir
        self.split_dir = split_dir
        self.transform = transform
        
        self.images_names = self._list_images_files()
    
    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.images_names[idx])
        image = rgb2lab(io.imread(image_path))
        image = image/(np.array([[116], [500], [200]])).reshape(1, 1, 3)
        
        image = image.astype(np.float32)

        label = self._label_of_image(self.images_names[idx])
        
        if(self.transform):
            image = self.transform(image)
        sample = (image, label)
        
        return sample

    def _label_of_image(self, image_name):
        if not isinstance(image_name, str):
            raise TypeError("Parameter image_name must be a string.")
        i = image_name.index("_")
        label = int(image_name[0:i]) - 1
        
        return label
    
    def _list_images_files(self):
        with open(self.split_dir, 'r') as f:
            filenames = f.read()
        
        return [filename for filename in filenames.split('\n') if len(filename) > 0]
        
    def weights_for_balance(self, nclasses):
        weights_dir = os.path.join(self.root_dir, '.weights-for-balance-{}.npy'.format(self.image_list))
        
        if(os.path.exists(weights_dir)):
            weight = np.load(weights_dir)
            return weight
        
        count = [0] * nclasses
        for image_name in self.images_names:
            label = self._label_of_image(image_name)
            count[label] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N/float(count[i])
        weight = [0] * self.__len__()
        for idx, image_name in enumerate(self.images_names):
            label = self._label_of_image(image_name)
            weight[idx] = weight_per_class[label]
        weight = np.array(weight, dtype=np.float)
        
        np.save(weights_dir, weight)
        
        return weight