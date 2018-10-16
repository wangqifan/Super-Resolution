import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from ImageDataset import ImageDataset






hr_transforms=transforms.Compose([ 
    #  transforms.Resize(256),
      transforms.ToTensor(), 
      transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
 ]) 
 
lr_transforms=transforms.Compose([ 
 #   transforms.Resize(64),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
 ]) 

dataloader=DataLoader(ImageDataset(lr_transforms=lr_transforms, hr_transforms=hr_transforms),
                        batch_size=256, shuffle=True, num_workers=4)