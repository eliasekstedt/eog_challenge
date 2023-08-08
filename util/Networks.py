
from datetime import datetime
import torch
import torch.nn as nn
from util.Special import RMSELoss
from util.Special import CustomLoss

class Res18FCNet(nn.Module):
    def __init__(self, architecture_name=None, weight_decay=0, dropout_rate=0, penalty=None):
        super(Res18FCNet, self).__init__()
        self.loss_fn = CustomLoss(penalty=penalty)
        self.s_loss_fn = RMSELoss()
        self.traincost, self.testcost = [], []
        self.s_traincost, self.s_testcost = [], []
        # architecture
        if architecture_name == 'res18fc':
            from util.Architectures import Res18FC
            self.architecture = Res18FC(dropout_rate)
        else:
            print('architecture not defined')
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

    def save_model(self, runpath):
        save_info = ''
        if len(self.s_testcost) >= 5:
            new_cost = self.s_testcost[-1]
            sorted_costs = sorted(self.s_testcost)
            if new_cost == sorted_costs[0]:
                torch.save(self.state_dict(), runpath+'model1.pth')
                save_info = f'\tsaved1!'
            elif new_cost == sorted_costs[1]:
                torch.save(self.state_dict(), runpath+'model2.pth')
                save_info = f'\tsaved2!'
            elif new_cost == sorted_costs[2]:
                torch.save(self.state_dict(), runpath+'model3.pth')
                save_info = f'\tsaved3!'
            elif new_cost == sorted_costs[3]:
                torch.save(self.state_dict(), runpath+'model4.pth')
                save_info = f'\tsaved4!'
            elif new_cost == sorted_costs[4]:
                torch.save(self.state_dict(), runpath+'model5.pth')
                save_info = f'\tsaved5!'
        return save_info

    def log_epoch(self, header, runpath, nr_epochs):
        current_epoch = len(self.testcost)
        epoch_info = f'{len(self.testcost)}/{nr_epochs}\t\t{round(self.traincost[-1], 4)}\t\t{round(self.testcost[-1], 4)}\t\t{round(self.s_traincost[-1], 4)}\t\t{round(self.s_testcost[-1], 4)}\t\t{str(datetime.now())[11:19]}'
        # save model if current best (in terms of test accuracy)
        #if self.s_testcost[-1] == min(self.s_testcost):
        #    torch.save(self.state_dict(), runpath+'model.pth')
        #    epoch_info = epoch_info + f'\tsaved!'
        save_info = ''
        if current_epoch > 10:#nr_epochs//2:
            save_info = f'\tsave{current_epoch}!'
            torch.save(self.state_dict(), runpath+f'model{current_epoch}.pth')
        epoch_info = epoch_info + save_info
        #save_info = self.save_model(runpath)
        # print and log current epoch info
        print(epoch_info)
        with open(runpath + 'log.txt', 'a') as file:
            if len(self.testcost) <= 1:
                file.write(header+'\n')
            file.write(epoch_info+'\n')

    def train_model(self, trainloader, testloader, nr_epochs, runpath, device):
        print(f'beginning training {str(datetime.now())[11:19]}')
        header = f'epoch\t\tCOST\t\tcost\t\tS_COST\t\ts_cost\t\ttime'
        print(header)
        for i in range(1, nr_epochs+1):
            self.train_epoch(trainloader, device)
            self.test_epoch(testloader, device)
            self.log_epoch(header, runpath, nr_epochs)































