import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
class FCModel(nn.Module):
    def __init__(self,n_in_features,n_out_features=4):
        super().__init__()
        self.n_in_features=n_in_features
        self.n_out_features=n_out_features
        self.net = nn.Sequential()
        self.net.add_module("fc1",nn.Linear(n_in_features,100))
        self.net.add_module("relu1",nn.ReLU(True))
        self.net.add_module("fc2",nn.Linear(100,70))
        self.net.add_module("relu2",nn.ReLU(True))
        self.net.add_module("fc3",nn.Linear(70,40))
        self.net.add_module("relu3",nn.ReLU(True))
        self.net.add_module("fc4",nn.Linear(40,n_out_features))
    def forward(self, input):
        output = self.net(input)
        return output
    def l2_regularization(self):
        loss=0
        total_num_params=0
        for k,layer in self.net._modules.items():
            if hasattr(layer,"weight"):
                loss=loss+torch.sum(layer.weight**2)
                total_num_params=total_num_params+np.array(layer.weight.shape).prod()
        loss=loss/total_num_params
        return loss
        