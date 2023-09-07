
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

    path = {'set_0':'WF1_classifier/csv/set_0.csv',
            'set_1':'WF1_classifier/csv/set_1.csv',
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
    
    with open('gp_test.txt', 'a') as file:
        file.write('\n#######################################')

    
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
    with open('eval_test.txt', 'a') as file:
        file.write(f'\n{accuracy}, {hparam["weight_decay"]}, {round(toc-tic, 4)}')
    return accuracy
    

def main():
    space = [Real(1e-6, 1e-5, name='wd')]
    result = gp_minimize(objective, space, n_calls=10, acq_func='EI', n_random_starts=5)
    print('optimization finished\n')
    best_param = result.x[0]
    print(result)
    print(f'best_param: {best_param}')



"""
to do:
***ALL OF THIS IM DOING BLINDLY***
*find optimal unrestricted crop ratio
*find optimal restricted crop ratio
*try other architectures: dualpathnet68, inception v3, NASNET-A-Large, unet(?) and see the intersection of misclassified images
*find out if they are similar in the optimal weight decay parameter (gaussian optimization)
*begin building v4 (regression): an assembly of three of these architectures and the context features.
    should they integrate all with each other or only the context features? this choise may have a great impact on the complexity of the model.
*implement mixed precision

always write down results from testing:
results are consistent for a large range of batch sizes and resolutions

in v4 or when using the full dataset:
*find optimal crop frequency
"""










if __name__ == '__main__':
    main()