



class JustRes(nn.Module):
    def __init__(self):
        print('using JustRes architecture')
        super(JustRes, self).__init__()
        nr_contextual_features = 16
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        nr_resnet18_infeatures = self.resnet18.fc.in_features
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1]) # removes the last layer
        self.fc1 = nn.Linear(nr_resnet18_infeatures+nr_contextual_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x, context):
        x = self.resnet18(x)
        x = x.view(x.size(0), -1)
        rescomponent = x.clone()
        x = torch.cat((x, context), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, rescomponent

    
class JustFC(nn.Module):
    def __init__(self):
        print('using JustFC architecture')
        super(JustFC, self).__init__()
        s0, s1, s2 = 528, 256, 128
        self.fc1 = nn.Linear(s0, s1)
        self.fc2 = nn.Linear(s1, s2)
        self.out = nn.Linear(s2, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    


class Res18_unintegrated(nn.Module):
    def __init__(self):
        print('using Res18_unintegrated architecture')
        super(Res18_unintegrated, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        nr_resnet18_infeatures = self.resnet18.fc.in_features
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1]) # removes the last layer
        self.fc1 = nn.Linear(nr_resnet18_infeatures, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x, context=torch.tensor([])):
        x = self.resnet18(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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