import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from model import Model
from DataLoader import dataloader



#Net=Model()

#Net=Net.cuda()
Net=torch.load("G.pkl")
criterion = nn.L1Loss()
criterion=criterion.cuda()

optimizer = optim.Adam(Net.parameters(), lr=0.0001)


epochs=100

for  epoch in range(epochs):
  for i, images in enumerate(dataloader):
        LR_image=Variable(images['lr'])
        H_images=Variable(images['hr']).cuda()


        LR_image=LR_image.cuda()

        optimizer.zero_grad()
        HR_images=Net(LR_image)
        loss=criterion(HR_images,H_images)
        loss.backward()
        optimizer.step()
        print("{:.6f}".format(loss.data[0]))
  print("-------------")
  torch.save(Net,"G.pkl")
    