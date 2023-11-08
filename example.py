
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

from visualize import visualize_predict

        
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device.type == "cuda":
    torch.cuda.set_device(0)
    
    
# name_model = 'vit_tiny'
name_model = 'vit_small'
# name_model = 'vit_base'
patch_size = 8

model = VitGenerator(name_model, patch_size, 
                     device, evaluate=True, random=False, verbose=True)
                     
path = 'corgi_image.jpg'
img = Image.open(path)
factor_reduce = 2
img_size = tuple(np.array(img.size[::-1]) // factor_reduce) 
print(f"corgi_image.jpg - img_size:{img_size}, patch_size:{patch_size}, type(img):{type(img)}")
visualize_predict(model, img, img_size, patch_size, device)

path = 'orange_cat.jpg'
img = Image.open(path)
factor_reduce = 2
img_size = tuple(np.array(img.size[::-1]) // factor_reduce) 
print(f"orange_cat.jpg - img_size:{img_size}, patch_size:{patch_size}, type(img):{type(img)}")
visualize_predict(model, img, img_size, patch_size, device)


