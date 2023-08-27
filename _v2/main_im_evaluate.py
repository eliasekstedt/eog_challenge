
import pandas as pd
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
    set_0 = Reader(path['fold_0'], path['data_labeled'], hparam['resizes'], eval=True)
    set_1 = Reader(path['fold_1'], path['data_labeled'], hparam['resizes'], eval=True)
    loader_0 = DataLoader(set_0, batch_size=hparam['batch_size'], shuffle=False)
    loader_1 = DataLoader(set_1, batch_size=hparam['batch_size'], shuffle=False)

    with open('run/_history/history.txt', 'r') as file:
        runpath = file.readlines()[-1]
    print(f'loading from {runpath}')

    from Networks import ImNet as Net
    model0 = Net('f0f1', hparam['architecture_name'], hparam['weight_decay'], hparam['penalty']).to(device)
    model0.load_state_dict(torch.load(runpath + model0.model_name + '.pth'))
    model0.eval()
    model1 = Net('f1f0', hparam['architecture_name'], hparam['weight_decay'], hparam['penalty']).to(device)
    model1.load_state_dict(torch.load(runpath + model1.model_name + '.pth'))
    model1.eval()

    from Evaluation_tools import ImEvaluator as Evaluator
    eve0 = Evaluator(runpath, model0, loader_1, path['fold_1'], device)
    eve1 = Evaluator(runpath, model1, loader_0, path['fold_0'], device)
    evaldata_assembly = pd.concat([eve0.evaldata, eve1.evaldata], axis=0)
    rmse = np.round(np.sqrt(np.mean((evaldata_assembly['extent'] - evaldata_assembly['pred'])**2)), 5)
    evaldata_assembly.to_csv(runpath+'evaldata.csv', index=False)
    print(f'rmse: {rmse}')

    from Evaluation_tools import Heatmap
    heatmap = Heatmap(runpath, height=110)
    heatmap.save(runpath)

    


if __name__ == '__main__':
    main()