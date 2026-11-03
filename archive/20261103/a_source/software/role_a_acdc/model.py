"""Frozen board-first 15-convolution topology; CAT is skip then nearest(deep)."""
from __future__ import annotations
import torch
from torch import nn
from .common import ARCHITECTURE

LAYER_NAMES=tuple([f'encoders.{i}.{j}' for i in range(4) for j in (0,2)]+
                  [f'decoders.{i}.{j}' for i in range(3) for j in (0,2)]+['head'])

class ACDCUNet(nn.Module):
    architecture=ARCHITECTURE
    def __init__(self):
        super().__init__()
        def block(ci,co):
            return nn.Sequential(nn.Conv2d(ci,co,3,padding=1),nn.ReLU(),
                                 nn.Conv2d(co,co,3,padding=1),nn.ReLU())
        self.encoders=nn.ModuleList([block(3,16),block(16,32),block(32,64),block(64,128)])
        self.pool=nn.MaxPool2d(2)
        self.decoders=nn.ModuleList([block(192,64),block(96,32),block(48,16)])
        self.head=nn.Conv2d(16,4,1)

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=3 or x.shape[-1]%8 or x.shape[-2]%8:
            raise ValueError('Expected NCHW three spatial channels and dimensions divisible by eight')
        skips=[]
        for i,block in enumerate(self.encoders):
            x=block(x);skips.append(x)
            if i<3:x=self.pool(x)
        for block,skip in zip(self.decoders,reversed(skips[:-1])):
            x=nn.functional.interpolate(x,scale_factor=2,mode='nearest')
            x=block(torch.cat((skip,x),dim=1))
        return self.head(x)

    def contract(self):
        layers=[(n,m) for n,m in self.named_modules() if isinstance(m,nn.Conv2d)]
        assert tuple(n for n,m in layers)==LAYER_NAMES
        assert sum(p.numel() for p in self.parameters())==487348
        return [{'id':f'C{i+1:02}','name':n,'in_channels':m.in_channels,
                 'out_channels':m.out_channels,'kernel':m.kernel_size[0],
                 'padding':m.padding[0],'relu':i<14} for i,(n,m) in enumerate(layers)]
