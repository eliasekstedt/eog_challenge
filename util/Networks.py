
from datetime import datetime
import torch
import torch.nn as nn

class ContextNet(nn.Module):
    def __init__(self, architecture_name=None, weight_decay=0, dropout_rate=0):
        super(ContextNet, self).__init__()
        self.loss_fn = nn.MSELoss() # try the RMSE in util.Special later
        self.traincost, self.testcost = [], []
        # architecture
        if architecture_name == 'res18_contextual':
            from util.Architectures import Res18_contextual
            self.architecture = Res18_contextual()
        else:
            print('architecture not defined')
            1/0
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=weight_decay)

    def forward(self, x, context):
        x = self.architecture(x, context)
        return x
    
    def train_epoch(self, trainloader, device):
        self.train()
        cost = 0
        for batch_images, batch_context, batch_labels, _ in trainloader: # loop iterations correspond to batches
            x, context, y = batch_images.to(device), batch_context.to(device), batch_labels.to(device)
            # prediction error
            preds = self(x, context)
            loss = torch.sqrt(self.loss_fn(preds, y))
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # record cost & accuracy
            cost += loss.item()/len(trainloader)
        self.traincost += [cost]

    def test_epoch(self, testloader, device):
        self.eval()
        cost = 0
        with torch.no_grad(): # disable gradient calculation
            for batch_images, batch_context, batch_labels, _ in testloader:
                x, context, y = batch_images.to(device), batch_context.to(device), batch_labels.to(device)
                preds = self(x, context)
                loss = torch.sqrt(self.loss_fn(preds, y))
                # record cost & accuracy
                cost += loss.item()/len(testloader)
        self.testcost += [cost]

    def log_epoch(self, header, runpath, nr_epochs):
        epoch_info = f'{len(self.testcost)}/{nr_epochs}\t\t{round(self.traincost[-1], 4)}\t\t{round(self.testcost[-1], 4)}\t\t{str(datetime.now())[11:19]}'
        # save model if current best (in terms of test accuracy)
        if self.testcost[-1] == min(self.testcost):
            torch.save(self.state_dict(), runpath+'model.pth')
            epoch_info = epoch_info + f'\tsaved!'
        # print and log current epoch info
        print(epoch_info)
        with open(runpath + 'log.txt', 'a') as file:
            if len(self.testcost) <= 1:
                file.write(header+'\n')
            file.write(epoch_info+'\n')

    def train_model(self, trainloader, testloader, nr_epochs, runpath, device):
        print(f'beginning training {str(datetime.now())[11:19]}')
        header = f'epoch\t\ttrain\t\ttest\t\ttime'
        print(header)
        for i in range(1, nr_epochs+1):
            self.train_epoch(trainloader, device)
            self.test_epoch(testloader, device)
            self.log_epoch(header, runpath, nr_epochs)
















class Orinet(nn.Module):
    def __init__(self, architecture_name=None, weight_decay=0, dropout_rate=0):
        super(Orinet, self).__init__()
        self.loss_fn = nn.MSELoss() # try the RMSE in util.Special later
        self.traincost, self.testcost = [], []
        # architecture
        if architecture_name == 'standard':
            from util.Architectures import Standard
            self.architecture = Standard(dropout_rate)
        elif architecture_name == 'res18_unintegrated':
            from util.Architectures import Res18_unintegrated
            self.architecture = Res18_unintegrated()
        else:
            print('architecture not defined')
            1/0
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=weight_decay)

    def forward(self, x):
        x = self.architecture(x)
        return x
    
    def train_epoch(self, trainloader, device):
        self.train()
        cost = 0
        for batch_images, batch_labels, _ in trainloader: # loop iterations correspond to batches
            x, y = batch_images.to(device), batch_labels.to(device)
            # prediction error
            preds = self(x)
            loss = torch.sqrt(self.loss_fn(preds, y))
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # record cost & accuracy
            cost += loss.item()/len(trainloader)
        self.traincost += [cost]

    def test_epoch(self, testloader, device):
        self.eval()
        cost = 0
        with torch.no_grad(): # disable gradient calculation
            for batch_images, batch_labels, _ in testloader:
                x, y = batch_images.to(device), batch_labels.to(device)
                preds = self(x)
                loss = torch.sqrt(self.loss_fn(preds, y))
                # record cost & accuracy
                cost += loss.item()/len(testloader)
        self.testcost += [cost]

    def log_epoch(self, header, runpath, nr_epochs):
        epoch_info = f'{len(self.testcost)}/{nr_epochs}\t{round(self.traincost[-1], 4)}\t{round(self.testcost[-1], 4)}\t{str(datetime.now())[11:19]}'
        # save model if current best (in terms of test accuracy)
        if self.testcost[-1] == min(self.testcost):
            torch.save(self.state_dict(), runpath+'model.pth')
            epoch_info = epoch_info + f'\tsaved!'
        # print and log current epoch info
        print(epoch_info)
        with open(runpath + 'log.txt', 'a') as file:
            if len(self.testcost) <= 1:
                file.write(header+'\n')
            file.write(epoch_info+'\n')

    def train_model(self, trainloader, testloader, nr_epochs, runpath, device):
        print(f'beginning training {str(datetime.now())[11:19]}')
        header = f'epoch\t\tt_cost\t\tv_cost\t\ttime'
        print(header)
        for i in range(1, nr_epochs+1):
            self.train_epoch(trainloader, device)
            self.test_epoch(testloader, device)
            self.log_epoch(header, runpath, nr_epochs)