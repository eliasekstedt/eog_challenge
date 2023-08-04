
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class Res18FC(nn.Module):
    def __init__(self, dropout_rate):
        print('using Res18FC architecture')
        super(Res18FC, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        nr_fc_in = self.resnet18.fc.in_features + 16
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1]) # removes the last layer
        self.fcblock = FCCBlock(nr_fc_in, dropout_rate)
        
    def forward(self, x, context):
        x = self.resnet18(x)
        x = x.view(x.size(0), -1)
        x = self.fcblock(torch.cat((x, context), dim=1))
        return x


class FCBlock(nn.Module):
    def __init__(self, nr_fc_in, dropout_rate):
        super(FCBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(nr_fc_in),
            nn.Linear(nr_fc_in, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.block(x)


"""
"""
class FCCBlock(nn.Module):
    def __init__(self, nr_fc_in, dropout_rate):
        print('using complex version FCCBlock')
        super(FCCBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(nr_fc_in),
            nn.Linear(nr_fc_in, nr_fc_in),
            nn.ReLU(),
            nn.BatchNorm1d(nr_fc_in),
            nn.Linear(nr_fc_in, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.block(x)

