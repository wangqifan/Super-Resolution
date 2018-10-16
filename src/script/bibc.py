from PIL import Image


import os
files=os.listdir("../image_hr_sub/")


for  file  in files:
    im = Image.open("../image_hr_sub/"+file)
    h,w=im.size
    imres=im.resize((h//4,w//4), Image.BICUBIC)
    imres.save("../image_lr_sub/"+file)