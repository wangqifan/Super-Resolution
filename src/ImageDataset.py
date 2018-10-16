import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self,lr_transforms, hr_transforms):
        self.lr_transform =lr_transforms
        self.hr_transform =hr_transforms

        self.files1 =sorted(os.listdir("./image_hr_sub/"))

        self.files2=sorted(os.listdir("./image_lr_sub/"))
    def __getitem__(self, index):
        img1 = Image.open("./image_hr_sub/"+self.files1[index % len(self.files1)])
        img2 = Image.open("./image_lr_sub/"+self.files2[index % len(self.files2)])
        img_hr = self.hr_transform(img1)
        img_lr = self.lr_transform(img2)
        return {'lr': img_lr, 'hr': img_hr}

    def __len__(self):
        return len(self.files1)