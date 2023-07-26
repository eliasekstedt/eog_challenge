
import torch
import torch.nn as nn

class Standard(nn.Module):
    def __init__(self, dropout_rate):
        print('using STANDARD architecture')
        super(Standard, self).__init__()
        self.coreblock0 = Coreblock(3, 8, dropout_rate)
        self.coreblock1 = Coreblock(8, 16, dropout_rate)
        self.endcore = CoreblockEnd(16, 32, dropout_rate)
        self.out = nn.Linear(32*32*32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.coreblock0(x)
        x = self.coreblock1(x)
        x = self.endcore(x)
        x = torch.flatten(x, 1)
        x = self.out(x)
        x = self.sigmoid(x)*100
        # no need for softmax here when using CEL. to get 'probabilities' in eval softmax can be applied there.
        return x
    
class Coreblock(nn.Module):
    def __init__(self, c_in, c_out, dropout_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.block(x)
    
class CoreblockEnd(nn.Module):
    def __init__(self, c_in, c_out, dropout_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.block(x)