
from datetime import datetime
import os
import torch
import torch.nn as nn


class Protocol:
    def __init__(self, hparam, model, penalty):
        self.criterion = nn.CrossEntropyLoss(weight=penalty)
        #self.criterion = nn.BCEWithLogitsLoss()
        self.traincost, self.valcost = [], []
        self.trainperformance, self.valperformance = [], []
        self.current_best = None
        self.model = model
        #self.optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=hparam['weight_decay'])

    def get_nr_accurate(self, logits, labels):
        preds = logits.argmax(1)
        evaluated_preds = (preds == labels)
        nr_accurate = evaluated_preds.sum()
        return nr_accurate.item()

    def train_epoch(self, trainloader, device):
        self.model.train()
        cost, performance = 0, 0
        for _, images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            cost += loss.item()/len(trainloader)
            performance += self.get_nr_accurate(logits, labels)/len(trainloader.dataset)
        self.traincost += [cost]
        self.trainperformance += [performance]

    def val_epoch(self, valloader, device):
        self.model.eval()
        cost, performance = 0, 0
        with torch.no_grad():
            for _, batch_images, batch_labels in valloader:
                images, labels = batch_images.to(device), batch_labels.to(device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                cost += loss.item()/len(valloader)
                performance += self.get_nr_accurate(logits, labels)/len(valloader.dataset)
        self.valcost += [cost]
        self.valperformance += [performance]

    def log_epoch(self, header, runpath, nr_epochs):
        epoch_info = f'{len(self.valcost)}/{nr_epochs}\t{round(self.traincost[-1], 4)}/{round(self.valcost[-1], 4)}\t{round(self.trainperformance[-1], 4)}/{round(self.valperformance[-1], 4)}\t{str(datetime.now())[11:19]}'
        if self.current_best is None or self.current_best >= self.valcost[-1]: # early stopping protocol
            self.current_best = self.valcost[-1]
            path_model = f"{runpath}model.pth"
            record_history(path_model)
            torch.save(self.model.state_dict(), path_model)
            epoch_info = f'{epoch_info} saved!'
        print(epoch_info)
        with open(runpath + 'log.txt', 'a') as file:
            if len(self.valcost) <= 1:
                file.write(header+'\n')
            file.write(epoch_info+'\n')
        
    def execute_protocol(self, trainloader, valloader, nr_epochs, runpath, device):
        print(f'beginning training {str(datetime.now())[11:19]}')
        header = f'epoch\tcost\t\tperformance\ttime'
        print(header)
        for i in range(1, nr_epochs+1):
            self.train_epoch(trainloader, device)
            self.val_epoch(valloader, device)
            self.log_epoch(header, runpath, nr_epochs)

def record_history(path_model):
    if not os.path.exists(path_model):
        with open('run/history/history.txt', 'a') as file:
            file.write(f"\n{path_model}")














