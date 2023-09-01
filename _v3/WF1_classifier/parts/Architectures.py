
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import warnings


class Architecture(nn.Module):
    def __init__(self):
        super(Architecture, self).__init__()
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.resnet = models.resnet18('ResNet18_Weights.DEFAULT')
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.resnet(x)
    