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
    ##visualize
    import numpy as np
    weight_path = 'D:\pixdot\CVTools\classification\model\Best.pt'
    model = classifier_model(num_class=2)
    model = model.to('cpu')
    chkpt = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(chkpt['model'])
    model.eval()
    from torchcam.methods import CAM,GradCAM,GradCAMpp,SmoothGradCAMpp,ScoreCAM,SSCAM,ISCAM,XGradCAM,LayerCAM
    from PIL import Image
    import torchvision.transforms as transforms
    from torchvision.models import resnet18
    from torchcam.methods import SmoothGradCAMpp
    from torchvision.transforms.functional import normalize, resize, to_pil_image   
    # Get your input
    import glob
    data_path = glob.glob("D:\pixdot\project_camera\data_two_classes\class2\*.bmp")
    img = Image.open(data_path[9])
    # Preprocess it for your chosen model
    to_tensor = transforms.Compose([            
                                        transforms.Resize([1024,1024]),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    
    input_tensor = to_tensor(img)
    # Preprocess your data and feed it to the model
    # Retrieve the CAM by passing the class index and the model output
    import matplotlib.pyplot as plt
    from torchcam.utils import overlay_mask
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,5))
    # for i,cam_extrac in enumerate([CAM,GradCAM,GradCAMpp,SmoothGradCAMpp,ScoreCAM,SSCAM,ISCAM,XGradCAM,LayerCAM]):
    cam_extractor = SmoothGradCAMpp(model)
    out = model(input_tensor.unsqueeze(0))
    _, predicted = torch.max(out.data,dim = 1)
    print(predicted)
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    # Visualize the raw CAM
    # plt.imshow(activation_map[0].squeeze(0).numpy());
    # plt.axis('off'); plt.tight_layout(); plt.show()

    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(np.array(img)), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    plt.axis('off'); 
    plt.imshow(result); 
    plt.tight_layout(); 
    plt.savefig('res.png',dpi=100)
    plt.show()