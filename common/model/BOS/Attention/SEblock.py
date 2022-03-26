# -*- encoding: utf-8 -*-
"""
@Time    : 2021/1/14 0:37
@Author  : Keith
@Software: PyCharm
"""
"""
SE

出自论文：Squeeze-and-Excitation Networks
论文链接：https://arxiv.org/pdf/1709.01507.pdf
是一种通道注意力机制。由于特征压缩和FC的存在，其捕获的通道注意力特征是具有全局信息的。

本文提出了一种新的结构单元——“Squeeze-and Excitation(SE)”模块，
可以自适应的调整各通道的特征响应值，对通道间的内部依赖关系进行建模。有以下几个步骤：
Squeeze: 沿着空间维度进行特征压缩，将每个二维的特征通道变成一个数，是具有全局的感受野。
Excitation: 每个特征通道生成一个权重，用来代表该特征通道的重要程度。
Reweight：将Excitation输出的权重看做每个特征通道的重要性，通过相乘的方式作用于每一个通道上。
"""
import torch
import torch.nn as nn
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上