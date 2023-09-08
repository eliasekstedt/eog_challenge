
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import warnings

class Architecture(nn.Module):
    def __init__(self):
        super(Architecture, self).__init__()
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.resnet(x)
    

"""
class Architecture(nn.Module):
    def __init__(self):
        super(Architecture, self).__init__()
        self.coreblock = self.Coreblock()
        self.outblock = self.Outblock()

    def forward(self, x):
        x = self.coreblock(x)
        return x
    
    class Coreblock(nn.Module):
        def __init__(self):
            self.block = nn.Conv2d(3, 1, 3)
        
        def forward(self, x):
            x = self.block(x)
            return x

    class Outblock(nn.Module):
        def __init__(self, nri, nro):
            self.block = nn.Linear(nri, nro)
        
        def forward(self, x):
            x = self.block(x)
            return x
"""