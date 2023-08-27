
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F



class Architecture(nn.Module):
    def __init__(self):
        print('using Im architecture')
        super(Architecture, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = torch.nn.Linear(num_ftrs, 1)

        
        
    def forward(self, x):
        return self.resnet18(x)
    