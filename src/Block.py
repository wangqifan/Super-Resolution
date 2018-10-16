import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self,n_feats,kernel_size,block_feats,wn,res_scale=1,act=nn.ReLU(True)):
        super(Block,self).__init__()
        self.res_scale=res_scale
        self.layers=nn.Sequential(
            wn(nn.Conv2d(n_feats,block_feats,kernel_size,padding=kernel_size//2)),
            act,
            wn(nn.Conv2d(block_feats,n_feats,kernel_size,padding=kernel_size//2))
        )
    def forward(self,x):
        res=self.layers(x)*self.res_scale
        res+=x
        return res