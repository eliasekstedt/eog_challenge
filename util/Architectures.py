
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F



class Nex(nn.Module):
    def __init__(self, dropout_rate):
        print('using NEX architecture')
        super(Nex, self).__init__()
        self.prenex_block = PreNexBlock(16, dropout_rate)
        # image
        self.resnet18 = models.resnet18(pretrained=True)
        nrexus = self.resnet18.fc.in_features + 128
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        # ...
        self.postnex_block = PostNexBlock(nrexus, dropout_rate)

        
    def forward(self, x, context):
        context = self.prenex_block(context)
        x = self.resnet18(x)
        x = x.view(x.size(0), -1)
        x = self.postnex_block(torch.cat((x, context), dim=1))
        return x

class PreNexBlock(nn.Module):
    def __init__(self, nr_in, dropout_rate):
        print('using PRENEXBLOCK')
        super(PreNexBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(nr_in, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 128),
        )
    
    def forward(self, x):
        return self.block(x)

class PostNexBlock(nn.Module):
    def __init__(self, nr_in, dropout_rate):
        print('using POSTNEXBLOCK')
        super(PostNexBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(nr_in, nr_in),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(nr_in, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.block(x)

class Res18FC(nn.Module):
    def __init__(self, dropout_rate):
        print('using Res18FC architecture')
        super(Res18FC, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
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


class VGG16(nn.Module):
    def __init__(self):
        print('using VGG16 architecture')
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16()
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, 1)
        
    def forward(self, x, context):
        x = self.vgg16(x)
        return x