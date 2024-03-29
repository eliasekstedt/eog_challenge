
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from WF1_classifier.parts.Architectures import Architecture

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
    def __init__(self, weight_decay, dropout_rate):
        super(Net, self).__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss() # eog_version
        #self.criterion = nn.CrossEntropyLoss() # dlia 3 unet-like version
        self.traincost, self.testcost = [], []
        self.trainaccuracy, self.testaccuracy = [], []
        self.record_performance = [0]
        # architecture
        self.architecture = Architecture(dropout_rate) # dlia3 version
        #self.architecture = Architecture() # eog_version
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=weight_decay)

    def forward(self, x):
        x = self.architecture(x)
        return x
    
    def train_epoch(self, trainloader, device):
        self.train()
        cost, accuracy = 0, 0
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
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # record cost & accuracy
            cost += loss.item()/len(trainloader)
            accuracy += self.get_nr_accurate(logits, labels)/len(trainloader.dataset)
        self.traincost += [cost]
        self.trainaccuracy += [accuracy]

    def get_nr_accurate(self, logits, labels, interpretability=False):
        if interpretability: # map +-inf to range 0-1. for uncertainty estimate and adjustable thresholds
            probs = F.sigmoid(logits)
            preds = (probs>0.5) * 1
        else:
            preds = (logits>0) * 1
        return (preds==labels).sum().item()

    def test_epoch(self, testloader, device):
        self.eval()
        cost, accuracy = 0, 0
        with torch.no_grad(): # disable gradient calculation
            for batch_images, batch_labels, _ in testloader:
                x, labels = batch_images.to(device), batch_labels.to(device)
                logits = self(x)
                loss = self.criterion(logits, labels)
                # record cost & accuracy
                cost += loss.item()/len(testloader)
                accuracy += self.get_nr_accurate(logits, labels)/len(testloader.dataset)
        self.testcost += [cost]
        self.testaccuracy += [accuracy]

    def log_epoch(self, header, runpath, nr_epochs):
        epoch_info = f'{len(self.testcost)}/{nr_epochs}\t\t{round(self.traincost[-1], 4)}/{round(self.trainaccuracy[-1], 4)}\t\t{round(self.testcost[-1], 4)}/{round(self.testaccuracy[-1], 4)}\t\t{str(datetime.now())[11:19]}'
        # save model if current best
        if min(self.trainaccuracy[-1], self.testaccuracy[-1]) >= max(self.record_performance):
            self.record_performance.append(min(self.trainaccuracy[-1], self.testaccuracy[-1]))
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
        header = f'epoch\t\ttr:cost/acc.\t\tte:cost/acc.\t\ttime'
        print(header)
        for i in range(1, nr_epochs+1):
            self.train_epoch(trainloader, device)
            self.test_epoch(testloader, device)
            self.log_epoch(header, runpath, nr_epochs)





