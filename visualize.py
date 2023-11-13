
# From
# https://github.com/aryan-jadon/Medium-Articles-Notebooks
# https://ai.plainenglish.io/visualizing-attention-in-vision-transformer-c871908d86de


# import os
import torch
import numpy as np
import math
# from functools import partial
import torch
import torch.nn as nn

import ipywidgets as widgets
import io
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

import warnings
warnings.filterwarnings("ignore")

from mingpt import VitGenerator


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
  
def transform(img, img_size):
    img = transforms.Resize(img_size)(img)
    img = transforms.ToTensor()(img)
    return img


def visualize_predict(model, img, img_size, patch_size, device):
    print(f"visualize_predict()")
    print(f"    img.size:{img.size}, img_size:{img_size}, patch_size:{patch_size}")
    img_pre = transform(img, img_size)
    attention = visualize_attention(model, img_pre, patch_size, device)
    # plot_attention(img, attention)


def visualize_attention(model, img, patch_size, device):
    print(f"visualize_attention()")
    print(f"    img.shape:{img.shape}, patch_size:{patch_size}, device:{device}")

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - \
        img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)
    print(f"    w:{w} h:{h} img.shape:{img.shape}")

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    print(f"    w_featmap:{w_featmap} h_featmap:{h_featmap}")

    attentions = model.get_last_selfattention(img.to(device))
    nh = attentions.shape[1]  # number of head
    print(f"    1)attentions.shape:{attentions.shape}, nh:{nh}")    # attentions.shape:torch.Size([1, 6, 9481, 9481]), nh:6

    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    print(f"    2)attentions.shape:{attentions.shape}")             # attentions.shape:torch.Size([6, 9480])

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    print(f"    3)attentions.shape:{attentions.shape}")             # attentions.shape:torch.Size([6, 79, 120])

    attentions = nn.functional.interpolate(attentions.unsqueeze(
        0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    print(f"    4)attentions.shape:{attentions.shape}")             attentions.shape:(6, 632, 960)

    return attentions


def plot_attention(img, attention):
    n_heads = attention.shape[0]

    plt.figure(figsize=(10, 10))
    text = ["Original Image", "Head Mean"]
    for i, fig in enumerate([img, np.mean(attention, 0)]):
        plt.subplot(1, 2, i+1)
        plt.imshow(fig, cmap='inferno')
        plt.title(text[i])
    plt.show()

    plt.figure(figsize=(10, 10))
    for i in range(n_heads):
        plt.subplot(n_heads//3, 3, i+1)
        plt.imshow(attention[i], cmap='inferno')
        plt.title(f"Head n: {i+1}")
    plt.tight_layout()
    plt.show()



class Loader(object):
    def __init__(self):
        self.uploader = widgets.FileUpload(accept='image/*', multiple=False)
        self._start()

    def _start(self):
        display(self.uploader)

    def getLastImage(self):
        try:
            for uploaded_filename in self.uploader.value:
                uploaded_filename = uploaded_filename
            img = Image.open(io.BytesIO(
                bytes(self.uploader.value[uploaded_filename]['content'])))

            return img
        except:
            return None

    def saveImage(self, path):
        with open(path, 'wb') as output_file:
            for uploaded_filename in self.uploader.value:
                content = self.uploader.value[uploaded_filename]['content']
                output_file.write(content)
