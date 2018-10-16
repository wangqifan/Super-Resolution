from PIL import Image
import random
import os
from  randomname import getrandomname





path="../DIV2K_train_HR/"
files=os.listdir(path)

for  file in files:
    im = Image.open(path+file)
    img_size =512
    x,y=im.size
    startx=0
    starty=0
    for i in range(int(x/img_size)):
        starty=0
        for j in  range(int(y/img_size)):
            region = im.crop((startx, starty, startx+img_size,starty+img_size))
            name=getrandomname()
            region.save("../image_hr_sub/"+name+".jpg")
            starty+=256
            print(name)
        startx+=img_size