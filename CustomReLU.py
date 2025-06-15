import torch.nn as nn
import torch

class CustomReLU(nn.Module):
    def __init__(self):
        super(CustomReLU,self).__init__()
    def forward(self,x):
        return torch.maximum(x,torch.tensor(0.0))