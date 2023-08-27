
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class FCBasic(nn.Module):
    def __init__(self, dropout_rate):
        print('using FC architecture')
        super(FCBasic, self).__init__()
        self.fcblock = FCBlock(16, 16, dropout_rate)
        
    def forward(self, context):
        context = self.fcblock(context)
        return context

class FCBlock(nn.Module):
    def __init__(self, n_in, n_core, dropout_rate):
        super(FCBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(n_in, n_core),
            nn.ReLU(),
            nn.Linear(n_core, n_core),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_core, 1)
        )
    
    def forward(self, x):
        return self.block(x)
    

class Im(nn.Module):
    def __init__(self):
        print('using Im architecture')
        super(Im, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 1)
        
    def forward(self, x):
        return self.resnet18(x)
    