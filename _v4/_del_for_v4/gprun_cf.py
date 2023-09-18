
import os
from datetime import datetime
import numpy as np
import torch

# make deterministic
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main():
    path = {'set_0':'WF1_classifier/csv/set_0.csv',
            'set_1':'WF1_classifier/csv/set_1.csv',
            'val':'WF1_classifier/csv/val.csv',
            'labeled':'../data/train/',
            'unlabeled':'../data/test/'
            }

    setup = {'tag':'cf',
             'hpq':'crop_freq',
             'bound': [0.0, 1.0],
             'n_calls': 2}
    
    hparam = {'batch_size': 64,
            'nr_epochs': 1,
            'weight_decay': 9.428542092781991e-05,
            'dropout_rate': 0.0,
            'augment_method': ['rcrop', 'hflip'],
            'crop_ratio': 0.5,
            'crop_freq': None,
            'usize': 128,
            'penalty': 1}


    gp_init_time = datetime.now()
    logpath = f'gp_logs/{setup["tag"]}_{str(gp_init_time)[8:10]}_{str(gp_init_time)[11:13]}_{str(gp_init_time)[14:16]}_{str(gp_init_time)[17:19]}/'
    if not os.path.isdir(logpath):
        os.makedirs(logpath)
    
    from WF1_classifier.parts.Optimizer import Optimizer
    opt = Optimizer(logpath, setup, path, hparam)
    opt.optimize()

    print('optimization finished')
    print(opt.rank)



if __name__ == '__main__':
    main()