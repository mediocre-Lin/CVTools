"""
@File: modelSet.py
@Author: Keith 
@Time: 2022/04/02 16:33:27
@Contact: 956744413@qq.com
"""
from .nets import FCN8s,Unet,SegNet
MODEL_DICT  = {
    'fcn8s':FCN8s,
    'unet':Unet,
    'segnet':SegNet
}
def segmentation_model(segNet,pre_trained = None, num_class = 2):
    net = MODEL_DICT[segNet](num_class = num_class)
    return net 
