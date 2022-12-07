# -*- code = utf-8 -*-
# @Time: 2022/11/4 16:54
# @Author: Chen Zigeng
# @File:CNN_network.py
# @Software:PyCharm
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class CA2Net(nn.Module):
    def __init__(self):
        super(CA2Net, self).__init__()

        self.cnn_layers = Sequential(

            Conv2d(1, 20, kernel_size=5),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(20, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

        )

        self.linear_layers = Sequential(
            Linear(1250, 26),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x



# # 实例化模型
# net = CA2Net()
#
# # 查看模型的各层的尺寸
# for name,param in net.named_parameters():
#     print(name, '-->', param.type(), '-->', param.dtype, '-->', param.shape)
#     #print(name,':',parameters.size())

