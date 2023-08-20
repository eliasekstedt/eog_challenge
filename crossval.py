
# external imports
import os
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip

# setting the seed
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main():
    # folderstructure setup
    path = {'fold0':'csv/fold0.csv',
            'fold1':'csv/fold1.csv',
            'valmap':'csv/Val.csv',
            'data_labeled':'data/train/',
            'data_unlabeled':'data/test/'
            }

    tag = 'crossval'
    device = 'cuda:0'

    # hyperparameters
    hparam = {'batch_size': 100,
            'nr_epochs': 3,
            'architecture_name':'res18fc',
            'weight_decay': 1e-7,
            'dropout_rate': 0.0,
            'resizes':(128, 128),
            'penalty': 1}

    # loading data
    from util.Readers import Res18FCReader as Reader
    fold0_set = Reader(path['fold0'], path['data_labeled'], resizes=hparam['resizes'], augment=True)
    fold0_loader = DataLoader(fold0_set, batch_size=hparam['batch_size'], shuffle=True)
    fold1_set = Reader(path['fold1'], path['data_labeled'], resizes=hparam['resizes'], augment=True)
    fold1_loader = DataLoader(fold1_set, batch_size=hparam['batch_size'], shuffle=True)

    # begin
    from util.Tools import run_init
    runpath = run_init(hparams=hparam, tag=tag, device=device)

    from util.Networks import Res18FCNet
    from util.Tools import performance_plot
    fold0_model = Res18FCNet(0, hparam['architecture_name'], hparam['weight_decay'], hparam['dropout_rate'], hparam['penalty']).to(device)
    fold0_model.train_model(trainloader=fold0_loader, testloader=fold0_loader, nr_epochs=hparam['nr_epochs'], runpath=runpath, device=device)
    performance_plot(fold0_model, runpath)
    fold1_model = Res18FCNet(0, hparam['architecture_name'], hparam['weight_decay'], hparam['dropout_rate'], hparam['penalty']).to(device)
    fold1_model.train_model(trainloader=fold1_loader, testloader=fold1_loader, nr_epochs=hparam['nr_epochs'], runpath=runpath, device=device)
    performance_plot(fold1_model, runpath)

    # generating submission file
    #from util.Readers import Res18FCReader as EvalReader
    #valset = EvalReader(path['valmap'], path['data_unlabeled'], resizes=hparam['resizes'])
    #valloader = DataLoader(valset, batch_size=hparam['batch_size'], shuffle=False)
    #network = Res18FCNet(hparam['architecture_name'], hparam['weight_decay'], hparam['dropout_rate']).to(device)

    #from util.Tools import create_submission
    #create_submission(network, runpath, valloader, hparam, device)
    #print('predictions generated, run finished\n')



if __name__ == '__main__':
    main()