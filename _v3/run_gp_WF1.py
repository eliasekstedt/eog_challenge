
import os
from datetime import datetime
import time
import numpy as np
import torch
import pandas

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

from skopt import gp_minimize
from skopt.space import Real

def objective(param):
    tag = f'WF1_{str(datetime.now())[8:10]}'
    #tag = tag + '/discard'

    path = {'set_0':'WF1_classifier/csv/set_0.csv',
            'set_1':'WF1_classifier/csv/set_1.csv',
            'val':'WF1_classifier/csv/val.csv',
            'labeled':'../data/train/',
            'unlabeled':'../data/test/'
            }

    hparam = {'batch_size': 64,
            'nr_epochs': 20,
            'weight_decay': param[0],
            'dropout_rate': 0.0,
            'augment_method': ['rcrop', 'hflip'],
            'crop_ratio': 0.5,
            'usize': 128,
            'penalty': 1}
    
    from WF1_classifier.Flow import Workflow
    workflow = Workflow(path=path, hparam=hparam, tag=tag)
    workflow.load_data()
    workflow.initiate_run()
    tic = time.perf_counter()
    workflow.learn_parameters()
    toc = time.perf_counter()
    workflow.evaluate()
    cm = workflow.evaluator.cmatrix
    accuracy = (cm[0,0] + cm[1,1])/cm.sum()
    with open('gp_wd_data.txt', 'a') as file:
        file.write(f'{accuracy}\t{hparam["weight_decay"]}\t{round(toc-tic)}\n')
    return accuracy
    

def main():
    l_bound, u_bound = 0, 1
    space = [Real(l_bound, u_bound, name='wd')]
    result = gp_minimize(objective, space, n_calls=25, acq_func='EI', n_random_starts=5)
    print('optimization finished\n')
    best_param = result.x[0]
    print(f'best_param: {best_param}')
    with open('gp_results.txt', 'a') as file:
        file.write(f'best_param: {best_param}')
    from WF1_classifier.parts.Tools import gp_plot
    gp_plot([l_bound, u_bound])
    




"""
consider broader extent and splitting into validation set as well
"""










if __name__ == '__main__':
    main()