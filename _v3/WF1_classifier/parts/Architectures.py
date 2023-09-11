
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import warnings

"""
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
    def __init__(self, dropout_rate):
        print('using dlia3 architecture')
        super(Architecture, self).__init__()
        self.block8 = self.Block8(3, 8)
        self.block16 = self.Block16(8, 16)
        self.block32 = self.Block32(16, 32)
        self.outblock = self.OutBlock(32*32*32, dropout_rate) #128 -> stridex2 -> 32, also 32 input channels

    def forward(self, x):
        x = self.block8(x)
        x = self.block16(x)
        x = self.block32(x)
        x = torch.flatten(x, 1)
        x = self.outblock(x)
        return x

    class Block8(nn.Module):
        def __init__(self, c_in, c_out):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, 1, 1), 
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        def forward(self, x):
            return self.block(x)
        
    class Block16(nn.Module):
        def __init__(self, c_in, c_out):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
    
        def forward(self, x):
            return self.block(x)
        
    class Block32(nn.Module):
        def __init__(self, c_in, c_out):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, 1, 1),
                nn.ReLU(),
            )
        
        def forward(self, x):
            return self.block(x)
        
    class OutBlock(nn.Module):
        def __init__(self, size, dropout_rate):
            super().__init__()
            self.block = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(size, 1)
            )
        
        def forward(self, x):
            return self.block(x)

