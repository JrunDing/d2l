"""
@Author: Jrun Ding
@Date: 2023.6.13
@Brief: 使用深度学习框架pytorch实现线性回归
@Coding: utf-8
"""

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn


def synthetic_data(w, b, num_examples):
    """生成y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


# 生成人工数据
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


# 1.调用框架中已有的API读取数据
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))


# 2.定义模型
net = nn.Sequential(nn.Linear(2, 1))


# 3.初始化训练参数
net[0].weight.data.normal_(0, 0.01)  # 初始化模型参数
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 4.训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

