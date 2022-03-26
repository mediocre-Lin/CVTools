"""
@File: ASPP.py
@Author: Keith 
@Time: 2022/03/26 14:49:35
@Contact: 956744413@qq.com
"""
# -*- encoding: utf-8 -*-
"""
@Time    : 2021/1/14 0:31
@Author  : Keith
@Software: PyCharm
"""
"""
ASPP
出自论文：DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Conv
论文链接：https://arxiv.org/pdf/1606.00915.pdf

空洞卷积的空间金字塔池化模块，主要是为了提高网络的感受野，并引入多尺度信息而提出的。
我们知道，对于语义分割网络，通常面临是分辨率较大的图片，这就要求我们的网络有足够的感受野来覆盖到目标物体。
对于CNN网络基本是靠卷积层的堆叠加上下采样操作来获取感受野的。
本文的该模块可以在不改变特征图大小的同时控制感受野，这有利于提取多尺度信息。
其中rate控制着感受野的大小，r越大感受野越大
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP,self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        # 不同空洞率的卷积
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
     # 池化分支
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear',align_corners = False)
     # 不同空洞率的卷积
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        # 汇合所有尺度的特征
        x = torch.cat([image_features, atrous_block1, atrous_block6,atrous_block12, atrous_block18], dim=1)
        # 利用1X1卷积融合特征输出
        x = self.conv_1x1_output(x)
        return x
