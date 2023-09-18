
from datetime import datetime
from Workflow.parts.lossFN import CustomLoss
from Workflow.parts.lossFN import RMSELoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from Workflow.parts.Architectures import Architecture

###

see = False
if see:
    import matplotlib.pyplot as plt
    def show(image, runpath='', title=''):
        #plt.imshow(image.to("cpu").permute(1, 2, 0))#, cmap='gray')
        plt.imshow(image.to("cpu").detach().permute(1, 2, 0))#, cmap='gray')
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        #plt.savefig(runpath+title+'.png')
        #plt.figure()
        #plt.close('all')
        plt.show()
###


class Net(nn.Module):
    def __init__(self, weight_decay, dropout_rate, penalty):
        super(Net, self).__init__()
        self.criterion = CustomLoss(penalty=penalty)
        self.s_criterion = RMSELoss()
        #self.criterion = torch.nn.BCEWithLogitsLoss() # eog_version
        #self.criterion = nn.CrossEntropyLoss() # dlia 3 unet-like version
        self.traincost, self.testcost = [], []
        self.s_traincost, self.s_testcost = [], []
        self.record_performance = []
        # architecture
        #self.architecture = Architecture(dropout_rate) # dlia3 version
        self.architecture = Architecture() # eog_version
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=weight_decay)

    def forward(self, x):
        x = self.architecture(x)
        return x
    
    def train_epoch(self, trainloader, device):
        self.train()
        cost, s_cost = 0, 0
        for batch_images, batch_labels, _ in trainloader: # loop iterations correspond to batches
            x, labels = batch_images.to(device), batch_labels.to(device)
            ###
            if see:
                """
                if '90_initial_1_1530_1530.JPG' in _:
                    ind = _.index('90_initial_1_1530_1530.JPG')
                    print(ind)
                    show(batch_images[ind])
                """
                print(batch_images.shape)
                for i in range(batch_images.shape[0]):
                    print(_[i])
                    show(batch_images[i])
            ###
            # prediction error
            logits = self(x)
            loss = self.criterion(logits, labels)
            s_loss = self.s_criterion(logits, labels)
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
            for batch_images, batch_labels, _ in testloader:
                x, labels = batch_images.to(device), batch_labels.to(device)
                logits = self(x)
                loss = self.criterion(logits, labels)
                s_loss = self.s_criterion(logits, labels)
                # record cost & accuracy
                cost += loss.item()/len(testloader)
                s_cost += (s_loss/len(testloader)).item()
        self.testcost += [cost]
        self.s_testcost += [s_cost]

    def log_epoch(self, header, runpath, nr_epochs):
        epoch_info = f'{len(self.testcost)}/{nr_epochs}\t\t{round(self.traincost[-1], 4)}/{round(self.s_traincost[-1], 4)}\t\t{round(self.testcost[-1], 4)}/{round(self.s_testcost[-1], 4)}\t\t{str(datetime.now())[11:19]}'
        # save model if current best
        if len(self.record_performance) == 0 or max(self.traincost[-1], self.testcost[-1]) <= min(self.record_performance):
            self.record_performance.append(min(self.traincost[-1], self.testcost[-1]))
            torch.save(self.state_dict(), f'{runpath}model.pth')
            epoch_info = epoch_info + f'\t*save*'
        # print and log current epoch info
        print(epoch_info)
        with open(runpath + 'log.txt', 'a') as file:
            if len(self.testcost) <= 1:
                file.write(header+'\n')
            file.write(epoch_info+'\n')

    def train_model(self, trainloader, testloader, nr_epochs, runpath, device):
        print(f'beginning training {str(datetime.now())[11:19]}')
        header = f'epoch\t\ttr:_c/sc.\t\tte:_c/sc.\t\ttime'
        print(header)
        for i in range(1, nr_epochs+1):
            self.train_epoch(trainloader, device)
            self.test_epoch(testloader, device)
            self.log_epoch(header, runpath, nr_epochs)





