"""
@File: SPP.py
@Author: Keith 
@Time: 2022/03/26 15:00:02
@Contact: 956744413@qq.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def SPP(previous_conv, num_sample, previous_conv_size, out_pool_size):
    """
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    """
    for i in range(len(out_pool_size)):
        h, w = previous_conv_size
        h_wid = math.ceil(h / out_pool_size[i])
        w_wid = math.ceil(w / out_pool_size[i])
        h_str = math.floor(h / out_pool_size[i])
        w_str = math.floor(w / out_pool_size[i])
        # print('(', h, w, ')')
        # print('window:', h_wid, w_wid)
        # print('stride:', h_str, w_str)
        max_pool = nn.MaxPool2d(kernel_size=(h_wid, w_wid), stride=(h_str, w_str))
        x = max_pool(previous_conv)
        if i == 0:
            spp = x.view(num_sample, -1)
        else:
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp
