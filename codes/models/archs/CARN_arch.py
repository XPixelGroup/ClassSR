import torch
import torch.nn as nn
import models.archs.arch_util as arch_util

class Block(nn.Module):
    def __init__(self, nf,
                 group=1):
        super(Block, self).__init__()

        self.b1 = arch_util.EResidualBlock(nf, nf, group=group)
        self.c1 = arch_util.BasicBlock(nf*2, nf, 1, 1, 0)
        self.c2 = arch_util.BasicBlock(nf*3, nf, 1, 1, 0)
        self.c3 = arch_util.BasicBlock(nf*4, nf, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
        

class CARN_M(nn.Module):
    def __init__(self, in_nc, out_nc, nf, scale=4, multi_scale=False, group=4):
        super(CARN_M, self).__init__()
        self.scale = scale
        rgb_range = 1
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = arch_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = arch_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.entry = nn.Conv2d(in_nc, nf, 3, 1, 1)

        self.b1 = Block(nf, group=group)
        self.b2 = Block(nf, group=group)
        self.b3 = Block(nf, group=group)
        self.c1 = arch_util.BasicBlock(nf*2, nf, 1, 1, 0)
        self.c2 = arch_util.BasicBlock(nf*3, nf, 1, 1, 0)
        self.c3 = arch_util.BasicBlock(nf*4, nf, 1, 1, 0)
        
        self.upsample = arch_util.UpsampleBlock(nf, scale=scale, 
                                          multi_scale=multi_scale,
                                          group=group)
        self.exit = nn.Conv2d(nf, out_nc, 3, 1, 1)
                
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=self.scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out