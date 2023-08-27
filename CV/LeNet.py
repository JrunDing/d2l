"""
@Author: Jrun Ding
@Date: 2023.8.27
@Brief: LeNet
@Coding: utf-8
"""

import torch
from torch import nn
from d2l import torch as d2l


class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


net = nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),  # 为了达到非线性性，原结构中没有
    nn.AvgPool2d(kernel_size=2, stride=2),  # 平均池化
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),  # 为了达到非线性性，原结构中没有
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),  # 第一维度保持，即batch维保持，其他维度展开
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

