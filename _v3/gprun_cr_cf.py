
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

    setup = {'tag':'cr_cf',                                             #
             'key_for_opt': ['crop_ratio', 'crop_freq'],                #
             'bounds': [(0.01, 0.99), (0, 1)],                          #
             'n_calls': 200}
    
    hparam = {'batch_size': 64,
            'nr_epochs': 20,
            'weight_decay': 9.428542092781991e-05,
            'dropout_rate': 0.0,
            'usize': 128,
            'penalty': 1,
            'method': ['rcrop', 'hflip'],
            'crop_ratio': None,                                           #
            'crop_freq': None}                                            #
    


    gp_init_time = datetime.now()
    logpath = f'gp_logs/{setup["tag"]}_{str(gp_init_time)[8:10]}_{str(gp_init_time)[11:13]}_{str(gp_init_time)[14:16]}_{str(gp_init_time)[17:19]}/'
    if not os.path.isdir(logpath):
        os.makedirs(logpath)
    
    from WF1_classifier.parts.Optimizer import Optimizer
    opt = Optimizer(logpath, setup, path, hparam)
    opt.optimize()

    print(f'optimization finished in {round(opt.rank["time"].sum()/3600, 2)} h')
    print(opt.rank)




"""
consider broader extent and splitting into validation set as well
best_param: 7.839123453947553e-05
"""





    #with open(f'{logpath}data.txt', 'a') as file:
    #    file.write(f'{accuracy}\t{param}\t{round(took)}\n')

    #with open(f'{logpath}data.txt', 'a') as file:
    #    file.write(f'\nbest_param: {best_param}')

"""
for multiple:
####################################
from skopt import gp_minimize

# Define the parameter space
space = [
    [0, 1],             # Binary parameter (0 or 1)
    (0.0, 1.0),         # Continuous parameter between 0 and 1
    (0.01, 0.99)        # Continuous parameter between 0.01 and 0.99
]

# Define the objective function
def objective(params):
    binary_param, continuous_param1, continuous_param2 = params
    # Your objective function here, which uses the parameters
    # ...
    return your_metric_here

# Perform the optimization
res = gp_minimize(objective, space)

"""


if __name__ == '__main__':
    main()