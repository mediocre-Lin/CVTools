"""
@File: model_visualize.py
@Author: Keith 
@Time: 2022/03/24 14:29:04
@Contact: 956744413@qq.com
"""
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from util import load_weight
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM, ISCAM, XGradCAM, LayerCAM
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
to_tensor = transforms.Compose([
    transforms.Resize([1024, 1024]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class ModelVision(object):
    def __init__(self, model, device='cpu') -> None:
        self.model = model
        self.device = device
        self.input = None
        self.cam_extractor = {
            'CAM': CAM,
            'GradCAM': GradCAM,
            'SmoothGradCAMpp': SmoothGradCAMpp,
            'ScoreCAM': ScoreCAM,
            'SSCAM': SSCAM,
            'ISCAM': ISCAM,
            'XGradCAM': XGradCAM,
            'LayerCAM': LayerCAM
        }

    def load_weight(self, weight_path):
        self.model = load_weight(self.model, weight_path, device=self.device)
    def visulize(self, img_path, method='GradCAM', trans=to_tensor,res_path = 'model_visulize.png'):
        plt.figure(figsize=(5, 5))
        img = Image.open(img_path)
        self.input = trans(img)
        cam_extract = self.cam_extractor[method]
        self.model.eval()
        out = self.model(self.input.unsqueeze(0))
        activation_map = cam_extract(out.squeeze(0).argmax().item(), out)
        res = overlay_mask(to_pil_image(np.array(img)), to_pil_image(
            activation_map[0].squeeze(0), mode='F'), alpha=0.5)
        plt.axis('off')
        plt.imshow(res)
        plt.tight_layout()
        plt.savefig(res_path, dpi=100)
        plt.show()
