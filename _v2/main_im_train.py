
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
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


def main():
    tag = 'test_im'
    device = 'cuda:0'

    path = {'fold_0':'csv/fold_0.csv',
            'fold_1':'csv/fold_1.csv',
            'valmap':'csv/Val.csv',
            'data_labeled':'data/train/',
            'data_unlabeled':'data/test/'
            }

    hparam = {'batch_size': 100,
            'nr_epochs': 15,
            'architecture_name':'im',
            'weight_decay': 1e-3,
            'dropout_rate': 0.0,
            'resizes':(128, 128),
            'penalty': 1}

    from Readers import ImReader as Reader
    set_0 = Reader(path['fold_0'], path['data_labeled'], hparam['resizes'], True)
    set_1 = Reader(path['fold_1'], path['data_labeled'], hparam['resizes'])
    loader_0 = DataLoader(set_0, batch_size=hparam['batch_size'], shuffle=True)
    loader_1 = DataLoader(set_1, batch_size=hparam['batch_size'], shuffle=False)

    from Tools import run_init
    runpath = run_init(hparams=hparam, tag=tag, device=device)

    from Networks import ImNet as Net
    model0 = Net('f0f1', hparam['architecture_name'], hparam['weight_decay'], hparam['penalty']).to(device)
    model1 = Net('f1f0', hparam['architecture_name'], hparam['weight_decay'], hparam['penalty']).to(device)

    model0.train_model(loader_0, loader_1, hparam['nr_epochs'], runpath, device)
    model1.train_model(loader_1, loader_0, hparam['nr_epochs'], runpath, device)

    from Tools import performance_plot
    performance_plot(model0, runpath)
    performance_plot(model1, runpath)
    


if __name__ == '__main__':
    main()