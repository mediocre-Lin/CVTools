"""
@File: modelSet.py
@Author: Keith 
@Time: 2022/03/23 16:00:48
@Contact: 956744413@qq.com
"""

import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

def calc_auto(num, channels):
    lst = [1, 2, 4, 8, 16, 32]
    return sum(map(lambda x: x ** 2, lst[:num])) * channels
class classifier_model(nn.Module):
    def __init__(self, model_cnn='resnet18',num_class = 4):
        super(classifier_model, self).__init__()
        self.model_cnn = model_cnn.lower()
        if self.model_cnn[:6] == 'resnet':
            model_conv = self.cnn_resnet()
            in_feature = list(model_conv.children())[-1].in_features
            model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
            model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        if 'efficientnet' in self.model_cnn[:-3]:
            model_conv = EfficientNet.from_pretrained(self.model_cnn)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            in_feature = list(model_conv.children())[-5].num_features
        if self.model_cnn[:5] == 'dense':
            model_conv = self.cnn_dense()
            in_feature = model_conv.classifier.in_features
            model_conv = nn.Sequential(*list(model_conv.children())[:-1],nn.ReLU(),nn.AdaptiveAvgPool2d(1))

        self.cnn = model_conv
        self.fc = nn.Linear(in_feature, num_class)


    def forward(self, img):
        if self.model_cnn[:-3] == 'efficientnet':
            feat = self.cnn.extract_features(img)
            feat = self.avgpool(feat)
        else:
            feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        out = self.fc(feat)

        return out
    def cnn_dense(self):
        if self.model_cnn == 'densenet121':
            return models.densenet121(pretrained=True)
        if self.model_cnn == 'densenet161':
            return models.densenet161(pretrained=True)
    def cnn_resnet(self):
        if self.model_cnn == 'resnet18':
            return models.resnet18(pretrained=True)
        if self.model_cnn == 'resnet34':
            return models.resnet34(pretrained=True)
        if self.model_cnn == 'resnet50':
            return models.resnet50(pretrained=True)
        if self.model_cnn == 'resnet101':
            return models.resnet101(pretrained=True)
        if self.model_cnn == 'resnet152':
            return models.resnet152(pretrained=True)


if __name__ == '__main__':
    pass
