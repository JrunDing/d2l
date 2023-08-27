"""
@Author: Jrun Ding
@Date: 2023.8.25
@Brief: Fashion-MNIST数据集处理，用于图像分类
@Coding: utf-8
"""
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()  # 用svg显示图像，清晰度较高

# 1.准备数据集
# 通过框架自带的内置函数将fashion-mnist数据集下载并读取到内存中
trans = transforms.ToTensor()  # 通过ToTensor实例将图像数据从PIL类型变换为32位浮点数格式
mnist_train = torchvision.datasets.FashionMNIST(root='./data/fashion_mnist', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root='./data/fashion_mnist', train=False, transform=trans, download=True)

print(len(mnist_train))  # 60000
print(type(mnist_train))  # <class 'torchvision.datasets.mnist.FashionMNIST'>
print(mnist_train[0][0].shape)  # 第0类的第0张图像的shape  torch.Size([1, 28, 28])


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))  # X是全部图像tensor格式，y是对应标签tensor([])
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

batch_size = 256


def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
print(type(train_iter))  # <class 'torch.utils.data.dataloader.DataLoader'>

for X, y in train_iter:
    continue


