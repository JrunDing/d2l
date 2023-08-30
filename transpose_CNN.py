"""
@Author: Jrun Ding
@Date: 2023.8.30
@Brief: 转置CNN
@Coding: utf-8
"""


import torch
from torch import nn
from d2l import torch as d2l

X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])  # 输入
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])  # 卷积核

# 默认2×2和2×2转置卷积 通道数为1
# tensor([[[[ 0.,  0.,  1.],
#           [ 0.,  4.,  6.],
#           [ 4., 12.,  9.]]]], grad_fn=<ConvolutionBackward0>)
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)  # 输入通道、输出通道、核大小、是否bias
tconv.weight.data = K
result = tconv(X)
print(result)

# 2×2和2×2转置卷积  通道数为1  padding为1
# tensor([[[[4.]]]], grad_fn=<ConvolutionBackward0>)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)  # 输入通道、输出通道、核大小、是否bias
tconv.weight.data = K
result = tconv(X)
print(result)

# 2×2和2×2转置卷积  通道数为1  stride为2
# tensor([[[[0., 0., 0., 1.],
#           [0., 0., 2., 3.],
#           [0., 2., 0., 3.],
#           [4., 6., 6., 9.]]]], grad_fn=<ConvolutionBackward0>)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)  # 输入通道、输出通道、核大小、是否bias
tconv.weight.data = K
result = tconv(X)
print(result)


# 16×16和16×16转置卷积  通道数为10
# torch.Size([1, 10, 16, 16])  发现参数一样，转置卷积的操作得到原始输入大小的tensor
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
print(tconv(conv(X)).shape)
