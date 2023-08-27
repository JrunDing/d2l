"""
@Author: Jrun Ding
@Date: 2023.8.25
@Brief: 多层感知机
@Coding: utf-8
"""

import torch
from torch import nn
from d2l import torch as d2l


# 1.加载数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 2.定义网络
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))


# 3.训练参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)  # 模型参数初始化
lr, num_epochs = 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)


# 4.开始训练
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
