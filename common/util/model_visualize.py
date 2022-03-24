"""
@File: model_visualize.py
@Author: Keith 
@Time: 2022/03/24 14:29:04
@Contact: 956744413@qq.com
"""
import numpy as np

class visualize(object):
    def __init__(self, model_info, method = 'CAM') -> None:
        self.model = model_info['model']
        self.final_conv = model_info['final_conv']
        self.features_blob = []
        self.net_name = []
        self.params = []
        def hook_feature(module, input, output):
            self.features_blobs.append(output.data.cpu().numpy())
        # 获取 features 模块的输出
        self.model._modules.get(self.final_conv).register_forward_hook(hook_feature)
        # 获取权重
        for name, param in self.model.named_parameters():
            self.net_name.append(name)
            self.params.append(param)
        self.weight_softmax = np.squeeze(self.params[-2].data.numpy())	# shape:(1000, 512)
        logit = self.model(img_variable)				# 计算输入图片通过网络后的输出值
        h_x = F.softmax(logit, dim=1).data.squeeze()	
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()					# 概率值排序
        idx = idx.numpy()						# 类别索引排序，概率值越高，索引越靠前



