
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

# ################################################################# #
# before real optimization, decide the best early stopping protocol #
# ################################################################# #
# this would involve a validation set and two different evaluators
def main():
    path = {'set_0':'Workflow/csv/set_0.csv',
            'set_1':'Workflow/csv/set_1.csv',
            'val':'Workflow/csv/set_1.csv',
            'labeled':'../data/train/',
            'unlabeled':'../data/test/'
            }

    setup = {'tag': '',                                          #
             'key_for_opt': ['weight_decay'],                #
             'bounds': [(0.0, 1e-5)],                      #
             'n_calls': 30} # 30
    
    hparam = {'batch_size': 64,
            'nr_epochs': 25, #25
            'weight_decay': None,
            'dropout_rate': 0.1,
            'usize': 128,
            'penalty': 1,
            'mode': 'res34',
            'method': ['hflip', 'rcrop'],
            'crop_ratio': 0.5,                                          #
            'crop_freq': 0.5}                                           #

    gp_init_time = datetime.now()
    logpath = f'gp_logs/{setup["tag"]}_{str(gp_init_time)[8:10]}_{str(gp_init_time)[11:13]}_{str(gp_init_time)[14:16]}_{str(gp_init_time)[17:19]}/'
    if not os.path.isdir(logpath):
        os.makedirs(logpath)
    
    from Workflow.parts.Optimizer import Optimizer
    opt = Optimizer(logpath, setup, path, hparam)
    opt.optimize()

    print(f'optimization finished in {round(opt.rank["time"].sum()/3600, 2)} h')
    print(opt.rank)




"""
best_param: 7.839123453947553e-05
"""



if __name__ == '__main__':
    main()