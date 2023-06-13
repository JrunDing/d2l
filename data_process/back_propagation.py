"""
@Author: Jrun Ding
@Date: 2023.6.13
@Brief: 反向传播
@Coding: utf-8
"""

# 假设我们想对函数y = 2x^T x  关于列向量x求导
import torch

x = torch.arange(4.0)

x.requires_grad_(True)  # 等价于 x = torch.arange(4.0, requires_grad=True)
# print(x.grad)  # 默认None，存储x的梯度值

y = 2*torch.dot(x, x)
y.backward()  # 4x
print(x.grad)

# 默认情况下，pytorch会累积梯度，我们需要清除之前的值
x.grad.zero_()

# 将某些计算移动到计算图之外，用于固定网络参数
y = x*x
y = y.detach()
z = u*x
z.sum().backward()

