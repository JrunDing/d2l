"""
@Author: Jrun Ding
@Date: 2023.8.25
@Brief: CNN
@Coding: utf-8
"""

import torch
from torch import nn
from d2l import torch as d2l


class Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)  # 2维卷积
        self.pool2d = nn.MaxPool2d(3, padding=1, stride=2)  # 最大池化层

    def forward(self, x):
        print(x.requires_grad)
        x = self.conv2d(x)
        x = self.pool2d(x)
        return x


#print(net.conv2d.weight)
#print(net.conv2d.bias)



