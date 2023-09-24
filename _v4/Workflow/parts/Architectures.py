
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import warnings

class Architecture(nn.Module):
    def __init__(self, dropout_rates, mode, fc_version):
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
        
        #nr_to_fc += 16
        if fc_version == 'wo1':
            print(f'fcv: {fc_version}')
            self.fcblock = FCBlock_wo1(nr_to_fc)
        elif fc_version == 'bn1':
            print(f'fcv: {fc_version}')
            self.fcblock = FCBlock_bn1(nr_to_fc)
        elif fc_version == 'wo2':
            print(f'fcv: {fc_version}')
            self.fcblock = FCBlock_wo2(nr_to_fc)
        elif fc_version == 'bn2':
            print(f'fcv: {fc_version}')
            self.fcblock = FCBlock_bn2(nr_to_fc)
        elif fc_version == 'wo3':
            print(f'fcv: {fc_version}')
            self.fcblock = FCBlock_wo3(nr_to_fc)
        elif fc_version == 'bn3':
            print(f'fcv: {fc_version}')
            self.fcblock = FCBlock_bn3(nr_to_fc)
        elif fc_version == 'no_fc':
            print(f'fcv: {fc_version}')
            self.fcblock = torch.nn.Linear(nr_to_fc, 1)

    def forward(self, x, context):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        #x = self.fcblock(torch.cat((x, context), dim=1))
        x = self.fcblock(x)
        return x
    
class FCBlock(nn.Module):
    def __init__(self, nr_fc_in, dropout_rates):
        super(FCBlock, self).__init__()
        self.block = nn.Sequential(
            #nn.Dropout(dropout_rates[0]),
            nn.Linear(nr_fc_in, 512),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            #nn.Dropout(dropout_rates[1]),
            #nn.Linear(256, 128),
            #nn.BatchNorm1d(256),
            #nn.ReLU(),
            #nn.Dropout(dropout_rates[2]),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.block(x)

class FCBlock_bn1(nn.Module):
    def __init__(self, nr_fc_in):
        super(FCBlock_bn1, self).__init__()
        print('FCBlock_bn1')
        self.block = nn.Sequential(
            nn.Linear(nr_fc_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.block(x)
    
class FCBlock_bn2(nn.Module):
    def __init__(self, nr_fc_in):
        super(FCBlock_bn2, self).__init__()
        print('FCBlock_bn2')
        self.block = nn.Sequential(
            nn.Linear(nr_fc_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.block(x)
    
class FCBlock_bn3(nn.Module):
    def __init__(self, nr_fc_in):
        super(FCBlock_bn3, self).__init__()
        print('FCBlock_bn3')
        self.block = nn.Sequential(
            nn.Linear(nr_fc_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.block(x)



class FCBlock_wo1(nn.Module):
    def __init__(self, nr_fc_in):
        super(FCBlock_wo1, self).__init__()
        print('FCBlock_wo1')
        self.block = nn.Sequential(
            nn.Linear(nr_fc_in, 512),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.block(x)
    
class FCBlock_wo2(nn.Module):
    def __init__(self, nr_fc_in):
        super(FCBlock_wo2, self).__init__()
        print('FCBlock_wo2')
        self.block = nn.Sequential(
            nn.Linear(nr_fc_in, 512),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.block(x)
    
class FCBlock_wo3(nn.Module):
    def __init__(self, nr_fc_in):
        super(FCBlock_wo3, self).__init__()
        print('FCBlock_wo3')
        self.block = nn.Sequential(
            nn.Linear(nr_fc_in, 512),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.block(x)
    