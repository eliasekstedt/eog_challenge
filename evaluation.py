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
    path = {'fold_0':'csv/fold0.csv',
            'fold_1':'csv/fold1.csv',
            'valmap':'csv/Val.csv',
            'data_labeled':'data/train/',
            'data_unlabeled':'data/test/'
            }

    device = 'cuda:0'

    # hyperparameters
    hparam = {'batch_size': 100,
            'nr_epochs': 12,
            'architecture_name':'res18fc',
            'weight_decay': 1e-7,
            'dropout_rate': 0.0,
            'resizes':(128, 128),
            'penalty': 1}
    

    #runpath = 'run/20_15_38_03_special/'
    runpath = 'run/21_07_37_42_special/'

    from util.Networks import Res18FCNet as Network

    # creating and loading model state here because could not be passed as a parameter for some reason
    models = [Network(fold=None, architecture_name=hparam['architecture_name'], weight_decay=hparam['weight_decay'], dropout_rate=hparam['dropout_rate'], penalty=hparam['penalty']).to(device), Network(fold=None, architecture_name=hparam['architecture_name'], weight_decay=hparam['weight_decay'], dropout_rate=hparam['dropout_rate'], penalty=hparam['penalty']).to(device)]
    for i, model in enumerate(models):
        model.load_state_dict(torch.load(runpath+'model_'+str(i)+'.pth'))
        model.eval()

    from util.Readers import Res18FCReader as Reader
    from util.Evaluators import CrossEvaluator as Evaluator
    Evaluator(runpath, models, Reader, path, hparam, device)











if __name__ == '__main__':
    main()





























