
from datetime import datetime
import torch
import torch.nn as nn
from WF1_classifier.parts.Architectures import Architecture

class Net(nn.Module):
    def __init__(self, architecture_name=None, weight_decay=0, penalty=None):
        super(Net, self).__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.traincost, self.testcost = [], []
        self.trainaccuracy, self.testaccuracy = [], []
        self.patience = None
        # architecture
        self.architecture = Architecture()
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=weight_decay)

    def forward(self, x):
        x = self.architecture(x)
        return x
    
    def train_epoch(self, trainloader, device):
        self.train()
        cost, accuracy = 0, 0
        for batch_images, batch_labels, _ in trainloader: # loop iterations correspond to batches
            x, y = batch_images.to(device), batch_labels.to(device)
            # prediction error
            preds = self(x)
            loss = self.criterion(preds, y)
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # record cost & accuracy
            cost += loss.item()/len(trainloader)
            accuracy += (preds.argmax(1)==y).sum().item()/len(trainloader.dataset)
        self.traincost += [cost]
        self.trainaccuracy += [accuracy]

    def test_epoch(self, testloader, device):
        self.eval()
        cost, accuracy = 0, 0
        with torch.no_grad(): # disable gradient calculation
            for batch_images, batch_labels, _ in testloader:
                x, y = batch_images.to(device), batch_labels.to(device)
                preds = self(x)
                loss = self.criterion(preds, y)
                # record cost & accuracy
                cost += loss.item()/len(testloader)
                accuracy += (preds.argmax(1)==y).sum().item()/len(testloader.dataset)
        self.testcost += [cost]
        self.testaccuracy += [accuracy]

    def log_epoch(self, header, runpath, nr_epochs):
        epoch_info = f'{len(self.testcost)}/{nr_epochs}\t\t{round(self.traincost[-1], 4)}/{round(self.trainaccuracy[-1], 4)}\t\t{round(self.testcost[-1], 4)}/{round(self.testaccuracy[-1], 4)}\t\t{str(datetime.now())[11:19]}'
        # save model if current best
        if len(self.testaccuracy) >= 8:
            if self.testaccuracy[-1] == min(self.testaccuracy[7:]):
                self.patience = 4
                torch.save(self.state_dict(), f'{runpath}model.pth')
                epoch_info = epoch_info + f'\tsaved!'
            else:
                self.patience -= 0 # infinite patience
        # print and log current epoch info
        print(epoch_info)
        with open(runpath + 'log.txt', 'a') as file:
            if len(self.testcost) <= 1:
                file.write(header+'\n')
            file.write(epoch_info+'\n')

    def train_model(self, trainloader, testloader, nr_epochs, runpath, device):
        print(f'beginning training {str(datetime.now())[11:19]}')
        header = f'epoch\t\ttr:cost/acc.\t\tte:cost/acc.\t\ttime'
        print(header)
        for i in range(1, nr_epochs+1):
            self.train_epoch(trainloader, device)
            self.test_epoch(testloader, device)
            self.log_epoch(header, runpath, nr_epochs)
            if self.patience is not None and self.patience <= 0:
                print('no more patience\n')
                break
            