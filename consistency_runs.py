
# external imports
import os
from torch.utils.data import DataLoader

# setting the seed
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main():
    #tag = 'testing500'
    tag = 'consistency_runs'
    device = 'cuda:0'

    # folderstructure setup
    path = {'trainmap':'csv/trainsplit.csv',
            'testmap':'csv/testsplit.csv',
            'valmap':'csv/Val.csv',
            'data_labeled':'data/train/',
            'data_unlabeled':'data/test/'
            }

    # hyperparameters
    hparam = {'batch_size': 128,
            'nr_epochs': 30,
            'architecture_name':'res18fc',
            'weight_decay': 1e-7,
            'dropout_rate': 0.0,
            'resizes': (128, 128),
            'penalty': 1}

    """
    # loading data
    """
    from util.Readers import Res18FCReader as Reader
    """
    #from util.Readers import SSCropReader as Reader
    trainset = Reader(path['trainmap'], path['data_labeled'], resizes=hparam['resizes'], augment=True)
    testset = Reader(path['testmap'], path['data_labeled'], resizes=hparam['resizes'])
    trainloader = DataLoader(trainset, batch_size=hparam['batch_size'], shuffle=True)
    testloader = DataLoader(testset, batch_size=hparam['batch_size'], shuffle=False)
    """

    if True:
        """
        # begin
        from util.Tools import run_init
        runpath = run_init(hparams=hparam, tag=tag, device=device)

        """
        runpath = 'run/consistency_runs/07_16_42_47/'
        from util.Networks import Res18FCNet
        model = Res18FCNet(hparam['architecture_name'], hparam['weight_decay'], hparam['dropout_rate'], hparam['penalty']).to(device)
        """
        #model.load_state_dict(torch.load('run/finding_weight_penalty/06_16_52_05/'+'model.pth'))
        model.train_model(trainloader, testloader, hparam['nr_epochs'], runpath, device)
        """

        # evaluation
        valset = Reader(path['valmap'], path['data_unlabeled'], resizes=hparam['resizes'], eval=True)
        testvalset = Reader(path['testmap'], path['data_labeled'], resizes=hparam['resizes'], eval=True)
        valloader = DataLoader(valset, batch_size=hparam['batch_size'], shuffle=False)
        testvalloader = DataLoader(testvalset, batch_size=hparam['batch_size'], shuffle=False)
        from util.Evaluation_tools import Evaluation
        Evaluation(hparam, model, testvalloader, valloader, runpath)

        from util.Tools import performance_plot
        performance_plot(model, runpath)


if __name__ == '__main__':
    main()