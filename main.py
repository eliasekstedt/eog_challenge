
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
    path = {'fold_0':'csv/trainsplit.csv',
            'fold_1':'csv/testsplit.csv',
            'valmap':'csv/Val.csv',
            'data_labeled':'data/train/',
            'data_unlabeled':'data/test/'
            }

    tag = 'nex_initiation'
    device = 'cuda:0'

    # hyperparameters
    hparam = {'batch_size': 100,
            'nr_epochs': 25,
            'architecture_name':'nex',
            'weight_decay': 5e-7,
            'dropout_rate': 0.0,
            'resizes':(128, 128),
            'penalty': 1}

    # loading data
    from util.Readers import Res18FCReader as Reader
    trainset = Reader(path['fold_0'], path['data_labeled'], resizes=hparam['resizes'], augment=True)
    testset = Reader(path['fold_1'], path['data_labeled'], resizes=hparam['resizes'])
    trainloader = DataLoader(trainset, batch_size=hparam['batch_size'], shuffle=True)
    testloader = DataLoader(testset, batch_size=hparam['batch_size'], shuffle=False)

    # begin
    from util.Tools import run_init
    runpath = run_init(hparams=hparam, tag=tag, device=device)

    from util.Networks import Res18FCNet
    model = Res18FCNet(0, hparam['architecture_name'], hparam['weight_decay'], hparam['dropout_rate'], hparam['penalty']).to(device)
    model.train_model(trainloader, testloader, hparam['nr_epochs'], runpath, device)

    # plot results
    from util.Tools import performance_plot
    performance_plot(model, runpath)




if __name__ == '__main__':
    main()