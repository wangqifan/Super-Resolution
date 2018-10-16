import torch
import torch.nn as nn
from Block import Block

from torch.nn.parameter import Parameter


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        scale=4
        n_resblocks = 8
        n_feats = 32
        kernel_size = 3
        act = nn.ReLU(True)

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.head=nn.Sequential(
            wn(nn.Conv2d(3, n_feats, 3, padding=0//2))
        )

        body = []
        for i in range(n_resblocks):
            body.append(
                Block(n_feats, kernel_size, n_feats, wn=wn, res_scale=4, act=act))
        self.body = nn.Sequential(*body)

        out_feats = scale*scale*3

        self.tail=nn.Sequential(
            wn(nn.Conv2d(n_feats, out_feats, 3, padding=0//2)),
            nn.PixelShuffle(scale)
        )

        self.skip=nn.Sequential(
            nn.Conv2d(3, out_feats, 5, padding=0),
            nn.PixelShuffle(scale)
        )

        self.padding=torch.nn.ReplicationPad2d(5//2)
    def forward(self, x):
 #       x = (x - self.rgb_mean.cuda()*255)/127.5
 #       if not self.training:
        x = self.padding(x)
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
 #       x = x*127.5 + self.rgb_mean.cuda()*255
        return x


