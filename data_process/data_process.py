"""
@Author: Jrun Ding
@Date: 2023.6.13
@Brief: torch.tensor数据结构的使用
@Coding: utf-8
"""

import torch


a = torch.tensor([1, 2, 3])
b = torch.arange(12)
print(b)  # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
print(b.shape)  # torch.Size([12])
print(b.numel())  # 12
print(b.reshape(3, 4))  # tensor([[ 0,  1,  2,  3],
                        #         [ 4,  5,  6,  7],
                        #         [ 8,  9, 10, 11]])
torch.zeros((2, 3, 4))  # 全0tensor
torch.ones((2, 3, 4))  # 全1tensor
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x+y)  # tensor([ 3.,  4.,  6., 10.])
print(x-y)  # tensor([-1.,  0.,  2.,  6.])
print(x*y)  # tensor([ 2.,  4.,  8., 16.])
print(x/y)  # tensor([0.5000, 1.0000, 2.0000, 4.0000])
print(x**y)  # tensor([ 1.,  4., 16., 64.])

x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
m = torch.cat((x, y), dim=0)
n = torch.cat((x, y), dim=1)

c = x.numpy()  # torch.tensor转numpy
d = torch.tensor(c)  # numpy转torch.tensor

e = torch.tensor([3.5])
e  # tensor([3.5000])
e.item()  # 3.5
float(e)  # 3.5
int(e)  # 3

