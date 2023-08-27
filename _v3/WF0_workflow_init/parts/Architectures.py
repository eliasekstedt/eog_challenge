
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F



class Architecture(nn.Module):
    def __init__(self):
        print('using Im architecture')
        super(Architecture, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 1)
        
    def forward(self, x):
        return self.resnet18(x)
    