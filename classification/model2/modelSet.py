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
        if self.model_cnn[:-3] == 'efficientnet':
            model_conv = EfficientNet.from_pretrained(self.model_cnn)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            in_feature = list(model_conv.children())[-5].num_features
        self.cnn = model_conv
        if self.use_spp == True:
            in_feature = calc_auto(3, in_feature*4)
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
  model = classifier_model(model_cnn='efficientnet-b1',use_spp=True)
  x = torch.rand((2,3,32,32))
  r = model(x)
  print(r[0].shape)
