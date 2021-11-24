# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from flim.experiments import utils
from flim.models.lcn import LCNCreator

import torch

import numpy as np

import matplotlib.pyplot as plt


# %%
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


# %%
architecture = utils.load_architecture("arch.json")


# %%
# the images are in LAB color space and have
images, markers = utils.load_images_and_markers("images_and_markers", normalize=False)


# %%
plt.imshow(utils.from_lab_to_rgb(images[0]))
indices = np.argwhere(markers[0] > 0)
plt.scatter(indices[:, 1], indices[:, 0], c="b", s=0.4)

# %%
plt.imshow(markers[0])


# %%
plt.imshow(utils.from_lab_to_rgb(images[0]))

x, y = np.where(markers[0] != 0)

plt.scatter(y, x, s=1, c=markers[0, x, y])


# %%
# relabel_markers=True will set a new label for each connected component in the markers
creator = LCNCreator(
    architecture, images=images, markers=markers, relabel_markers=False, device=device
)


# %%
# Build the feature extractor using FLIM
creator.build_model()


# %%
# model is a PyTorch Module https://pytorch.org/docs/stable/generated/torch.nn.Module.html
model = creator.get_LIDSConvNet()


# %%
# input mut be a PyTorch Tensor with shape (N, C, H, W)
x = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)


# %%
with torch.no_grad():
    features = model.forward(x)


# %%
print(features.size())


# %%
plt.imshow(features[0, 16, :, :].cpu())


# %%


# %%
