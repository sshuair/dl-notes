#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : sshuair
# @Time    : 2017/4/29
# @Project : kaggle-planet

from functools import reduce
import torch.nn as nn
import torch.nn.functional as F


class LenNt(nn.Module):
    def __init__(self, num_classes):
        super(LenNt, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16*61*61, 100)  # TODO: 生成方法，不要手动的计算这个数字
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x):
        """

        :param x: 一个batch的数据，输入结构：[batch_size, image_channel, image_width, image_height]
        :return: 前向传播结果
        """
        # 第一阶段特征提取（卷积、relu、池化）
        x = self.pool1(F.relu(self.conv1(x)))

        # 第二阶段特征提取（卷积、relu、池化）
        x = self.pool2(F.relu(self.conv2(x)))

        # dropout
        x = self.drop(x)

        # flat features 多维维空间一维化，输出也就是[batch_size, image_channel * image_width * image_height]
        x = x.view(-1, self.num_flat_features(x))
        # x = x.view(batch_size, -1)  不能用这种方法，因为最后一个epoch可能不是正好整除batch_size

        # 全连接层（fc1，fc2，fc3）
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # return F.sigmoid(x)
        return x
        

    def num_flat_features(self, x):
        feature_size = x.size()[1:]
        num_flat = reduce(lambda m, n: m * n, feature_size, 1)
        return num_flat



class PlaentNet(nn.Module):
    def __init__(self, num_classes):
        super(PlaentNet, self).__init__()
        self.features = nn.Sequential(
            # input_shape=[batch_size, 3, 32, 32] batch_size, image_channel, image_width, image_height
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(12544,128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # self.pool1 = nn.MaxPool2d(2)
        # self.drop1 = nn.Dropout2d(0.25)
        # self.fc1 = nn.Linear(2304, 128)  # TODO: 生成方法，不要手动的计算这个数字
        # self.drop2 = nn.Dropout2d(0.5)
        # self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        :param x: 一个batch的数据，输入结构：[batch_size, image_channel, image_width, image_height]
        :return: 前向传播结果
        """
        # # 第一阶段特征提取（卷积、relu、池化）
        # x = self.pool1(F.relu(self.conv1(x)))

        # # 第二阶段特征提取（卷积、relu、池化）
        # x = self.pool2(F.relu(self.conv2(x)))

        # # dropout
        # x = self.drop1(x)

        # # flat features 多维维空间一维化，输出也就是[batch_size, image_channel * image_width * image_height]
        # x = x.view(-1, self.num_flat_features(x))
        # # x = x.view(batch_size, -1)  不能用这种方法，因为最后一个epoch可能不是正好整除batch_size

        # # 全连接层（fc1，fc2，fc3）
        # x = F.relu(self.fc1(x))
        # x = self.drop2(x)

        # x = F.relu(self.fc2(x))

        # x = F.sigmoid(x)
        x = self.features(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.classifier(x)
        return x

    def num_flat_features(self, x):
        feature_size = x.size()[1:]
        num_flat = reduce(lambda m, n: m * n, feature_size, 1)
        return num_flat
