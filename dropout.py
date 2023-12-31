"""
@Author: Jrun Ding
@Date: 2023.8.25
@Brief: dropout层
@Coding: utf-8
"""

import torch
from torch import nn
from d2l import torch as d2l

# 1.准备数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 2.定义模型
dropout1, dropout2 = 0.2, 0.5

net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))


# 3.训练参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)
num_epochs, lr = 10, 0.5
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# 4.开始训练
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
