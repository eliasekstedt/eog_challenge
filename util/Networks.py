
from datetime import datetime
import torch
import torch.nn as nn
from util.Special import RMSELoss
from util.Special import CustomLoss

class Res18FCNet(nn.Module):
    def __init__(self, fold=None, architecture_name=None, weight_decay=0, dropout_rate=0, penalty=None):
        super(Res18FCNet, self).__init__()
        self.fold = fold
        self.loss_fn = CustomLoss(penalty=penalty)
        self.s_loss_fn = RMSELoss()
        self.traincost, self.testcost = [], []
        self.s_traincost, self.s_testcost = [], []
        self.patience = None
        # architecture
        if architecture_name == 'nex':
            from util.Architectures import Nex
            self.architecture = Nex(dropout_rate)
        elif architecture_name == 'res18fc':
            from util.Architectures import Res18FC
            self.architecture = Res18FC(dropout_rate)
        elif architecture_name == 'vgg16':
            from util.Architectures import VGG16
            self.architecture = VGG16()
        else:
            print(f'architecture {architecture_name} is not defined')
            1/0
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=weight_decay)

    def forward(self, x, context):
        x = self.architecture(x, context)
        return x
    
    def train_epoch(self, trainloader, device):
        self.train()
        cost, s_cost= 0, 0
        for batch_images, batch_context, batch_labels, _ in trainloader: # loop iterations correspond to batches
            x, context, y = batch_images.to(device), batch_context.to(device), batch_labels.to(device)
            # prediction error
            preds = self(x, context)
            loss = self.loss_fn(preds, y)
            s_loss = self.s_loss_fn(preds, y)
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # record cost & accuracy
            cost += loss.item()/len(trainloader)
            s_cost += (s_loss/len(trainloader)).item()
        self.traincost += [cost]
        self.s_traincost += [s_cost]

    def test_epoch(self, testloader, device):
        self.eval()
        cost, s_cost = 0, 0
        with torch.no_grad(): # disable gradient calculation
            for batch_images, batch_context, batch_labels, _ in testloader:
                x, context, y = batch_images.to(device), batch_context.to(device), batch_labels.to(device)
                preds = self(x, context)
                loss = self.loss_fn(preds, y)
                s_loss = self.s_loss_fn(preds, y)
                # record cost & accuracy
                cost += loss.item()/len(testloader)
                s_cost += (s_loss/len(testloader)).item()
        self.testcost += [cost]
        self.s_testcost += [s_cost]

    def log_epoch(self, header, runpath, nr_epochs):
        epoch_info = f'{len(self.testcost)}/{nr_epochs}\t\t{round(self.traincost[-1], 4)}\t\t{round(self.testcost[-1], 4)}\t\t{round(self.s_traincost[-1], 4)}\t\t{round(self.s_testcost[-1], 4)}\t\t{str(datetime.now())[11:19]}'
        # save model if current best
        if len(self.s_testcost) >= 8:
            if self.s_testcost[-1] == min(self.s_testcost[7:]):
                self.patience = 4
                torch.save(self.state_dict(), runpath+'model_'+str(self.fold)+'.pth')
                epoch_info = epoch_info + f'\tsaved!'
            else:
                self.patience -= 1
        # print and log current epoch info
        print(epoch_info)
        with open(runpath + 'log.txt', 'a') as file:
            if len(self.testcost) <= 1:
                file.write(header+'\n')
            file.write(epoch_info+'\n')

    def train_model(self, trainloader, testloader, nr_epochs, runpath, device):
        print(f'beginning training fold_{self.fold} {str(datetime.now())[11:19]}')
        header = f'epoch\t\tCOST\t\tcost\t\tS_COST\t\ts_cost\t\ttime'
        print(header)
        for i in range(1, nr_epochs+1):
            self.train_epoch(trainloader, device)
            self.test_epoch(testloader, device)
            self.log_epoch(header, runpath, nr_epochs)
            if self.patience is not None and self.patience <= 0:
                print('no more patience\n')
                break
            
