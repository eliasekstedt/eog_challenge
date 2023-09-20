
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import warnings

class Architecture(nn.Module):
    def __init__(self, dropout_rates, mode):
        super(Architecture, self).__init__()
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        #self.conv = models.resnext50_32x4d(pretrained=True)
        if mode == 'res18':
            self.conv = models.resnet18(pretrained=True)
            nr_to_fc = self.conv.fc.in_features
            self.conv = nn.Sequential(*list(self.conv.children())[:-1]) # removes the last layer
            #self.resnet.fc = torch.nn.Linear(nr_fc_in, 1)
        elif mode == 'res34':
            self.conv = models.resnet34(pretrained=True)
            nr_to_fc = self.conv.fc.in_features
            self.conv = nn.Sequential(*list(self.conv.children())[:-1]) # removes the last layer
        elif mode == 'mobv2':
            self.conv = models.mobilenet_v2(pretrained=True, progress=True)
            nr_to_fc = self.conv.classifier[1].in_features
            self.conv.classifier = nn.Identity()
        
        nr_to_fc += 16
        self.fcblock = FCBlock(nr_to_fc, dropout_rates)

    def forward(self, x, context):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fcblock(torch.cat((x, context), dim=1))
        return x
    
class FCBlock(nn.Module):
    def __init__(self, nr_fc_in, dropout_rates):
        super(FCBlock, self).__init__()
        self.block = nn.Sequential(
            #nn.BatchNorm1d(nr_fc_in),
            nn.Dropout(dropout_rates[0]),
            nn.Linear(nr_fc_in, 256),
            nn.ReLU(),
            #nn.BatchNorm1d(256),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(256, 128),
            nn.ReLU(),
            #nn.BatchNorm1d(128),
            nn.Dropout(dropout_rates[2]),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.block(x)
