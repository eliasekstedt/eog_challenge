
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import warnings

"""
"""
class Architecture(nn.Module):
    def __init__(self, dropout_rate):
        super(Architecture, self).__init__()
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        #self.conv = models.resnext50_32x4d(pretrained=True)
        self.conv = models.resnet18(pretrained=True)
        nr_fc_in = self.conv.fc.in_features + 8


        self.conv = nn.Sequential(*list(self.conv.children())[:-1]) # removes the last layer
        self.fcblock = FCBlock(nr_fc_in, dropout_rate)

        #self.resnet.fc = torch.nn.Linear(num_ftrs, 1)

    def forward(self, x, context):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fcblock(torch.cat((x, context), dim=1))
        return x
    
class FCBlock(nn.Module):
    def __init__(self, nr_fc_in, dropout_rate):
        super(FCBlock, self).__init__()
        self.block = nn.Sequential(
            #nn.BatchNorm1d(nr_fc_in),
            nn.Linear(nr_fc_in, 256),
            nn.ReLU(),
            #nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            #nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.block(x)

"""
class Architecture(nn.Module):
    def __init__(self):
        print('Using SC architecture')
        super(Architecture, self).__init__()
        self.short0 = self.Shortblock(3, 64)
        self.narro0 = self.Narblock(64, 64)
        self.short1 = self.Shortblock(64, 128)
        self.narro1 = self.Narblock(128, 128)
        self.short2 = self.Shortblock(128, 256)
        self.narro2 = self.Narblock(256, 256)
        self.short3 = self.Shortblock(256, 512)
        self.narro3 = self.Narblock(512, 512)
        self.short4 = self.Shortblock(512, 1024)
        # U middle U
        self.widen0 = self.Widblock(1024, 512)
        self.short5 = self.Shortblock(1024, 512)
        self.widen1 = self.Widblock(512, 256)
        self.short6 = self.Shortblock(512, 256)
        self.widen2 = self.Widblock(256, 128)
        self.short7 = self.Shortblock(256, 128)
        self.widen3 = self.Widblock(128, 64)
        self.short8 = self.Shortblock(128, 64)
        self.short9 = self.Shortblock(64, 1)

    def forward(self, x):
        x0 = self.short0(x)
        x1 = self.narro0(x0)
        x1 = self.short1(x1)
        x2 = self.narro1(x1)
        x2 = self.short2(x2)
        x3 = self.narro2(x2)
        x3 = self.short3(x3)
        xx = self.narro3(x3)
        xx = self.short4(xx)
        # U middle U
        xx = self.widen0(xx)
        xx = self.short5(torch.cat([x3, xx], dim=1))
        xx = self.widen1(xx)
        xx = self.short6(torch.cat([x2, xx], dim=1))
        xx = self.widen2(xx)
        xx = self.short7(torch.cat([x1, xx], dim=1))
        xx = self.widen3(xx)
        xx = self.short8(torch.cat([x0, xx], dim=1))
        xx = self.short9(xx)
        return torch.sigmoid(xx)

    class Narblock(nn.Module): # pixelwise narrow
        def __init__(self, c_in, c_out):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        def forward(self, x):
            return self.block(x)

    class Widblock(nn.Module): # pixelwise widen
        def __init__(self, c_in, c_out):
            super().__init__()
            self.block = nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)
        
    class Shortblock(nn.Module): # channelwise lengthen with batch normalization
        def __init__(self, c_in, c_out):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )
        
        def forward(self, x):
            return self.block(x)

"""

"""
class Architecture(nn.Module):
    def __init__(self, dropout_rate):
        print('using dlia3 architecture')
        super(Architecture, self).__init__()
        self.block8 = self.Block8(3, 8)
        self.block16 = self.Block16(8, 16)
        self.block32 = self.Block32(16, 32)
        self.outblock = self.OutBlock(32*32*32, dropout_rate) #128 -> stridex2 -> 32, also 32 input channels
        ###
        self.catblock = self.CatBlock(2*32*32*32, dropout_rate)

    def forward(self, x):
        a = self.block8(x)
        a = self.block16(a)
        a = self.block32(a)
        a = torch.flatten(a, 1)
        #a = self.outblock(a)
        ###
        b = self.block8(x)
        b = self.block16(b)
        b = self.block32(b)
        b = torch.flatten(b, 1)
        #b = self.outblock(b)

        ###
        x = torch.cat([a, b], dim=1) # its not a good solution...
        x = self.catblock(x)
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

    class CatBlock(nn.Module):
        def __init__(self, size, dropout_rate):
            super().__init__()
            self.block = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(size, 1)
            )
        
        def forward(self, x):
            return self.block(x)

"""

