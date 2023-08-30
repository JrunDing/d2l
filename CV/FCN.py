"""
@Author: Jrun Ding
@Date: 2023.8.30
@Brief: 全连接卷积神经网络 FCN
@Coding: utf-8
"""


import torch
import torchvision
from torch import nn


pretained_net = torchvision.models.resnet18(pretrained=True)
# list(pretained_net.children())[-3, :]

net = nn.Sequential(*list(pretained_net.children())[:-2])  # 去掉网络最后两层
X = torch.rand(size=(1, 3, 320, 480))

num_classes = 21  # VOC2012共21类
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32))


