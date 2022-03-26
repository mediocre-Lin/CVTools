# -*- encoding: utf-8 -*-
"""
@Time    : 2021/1/14 0:35
@Author  : Keith
@Software: PyCharm
"""
"""
Non-local
出自论文：Non-local Neural Networks
论文链接：https://arxiv.org/abs/1711.07971

Non-Local是一种attention机制，也是一个易于植入和集成的模块。
Local主要是针对感受野(receptive field)来说的，
以CNN中的卷积操作和池化操作为例，它的感受野大小就是卷积核大小，
而我们常用3X3的卷积层进行堆叠，它只考虑局部区域，都是local的运算。
不同的是，non-local操作感受野可以很大，可以是全局区域，而不是一个局部区域。
捕获长距离依赖（long-range dependencies），即如何建立图像上两个有一定距离的像素之间的联系，是一种注意力机制。
所谓注意力机制就是利用网络生成saliency map，注意力对应的是显著性区域，是需要网络重点关注的区域。
"""
import torch
import torch.nn as nn
class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(channel, self.inter_channel, 1, 1,0, False)
        self.conv_theta = nn.Conv2d(channel, self.inter_channel, 1, 1,0, False)
        self.conv_g = nn.Conv2d(channel, self.inter_channel, 1, 1, 0, False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(self.inter_channel, channel, 1, 1, 0, False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # 获取phi特征，维度为[N, C/2, H * W]，注意是要保留batch和通道维度的，是在HW上进行的
        x_phi = self.conv_phi(x).view(b, c, -1)
        # 获取theta特征，维度为[N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # 获取g特征，维度为[N, H * W, C/2]
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # 对phi和theta进行矩阵乘，[N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        # softmax拉到0~1之间
        mul_theta_phi = self.softmax(mul_theta_phi)
        # 与g特征进行矩阵乘运算，[N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # 1X1卷积扩充通道数
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x # 残差连接
        return out