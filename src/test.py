import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.transforms  as transforms 
from torchvision.utils import save_image

from PIL import Image



SRgen=torch.load("G.pkl")
loader=transforms.Compose([
    transforms.ToTensor()
])


def imageloader(image_name):
    image=Image.open(image_name)
    image=Variable(loader(image))
    image=image.unsqueeze(0)
    return image


imagename="input.png"

image=imageloader(imagename)

image=image.cuda()
image=SRgen(image)
newname="temp.jpg"
image=image.cpu()
save_image(image.data,newname)
