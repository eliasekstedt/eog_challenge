
import torch.nn as nn
#import torchvision.models as models
#import warnings
import timm
import torch.nn.functional as F

class CMOS31(nn.Module):
    def __init__(self):
        super(CMOS31, self).__init__()
        self.model = timm.create_model('resnet50', pretrained=False)
        nr_features = self.model.fc.in_features
        self.model.fc = nn.Linear(nr_features, 2)

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)














































