import torch
import numpy as np


def init_layer(layer, gain=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



   
