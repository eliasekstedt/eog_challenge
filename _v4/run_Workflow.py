
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

#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main():
    tag = f'WF1_{str(datetime.now())[8:10]}_'
    tag = tag + 'testing'

    path = {'set_0':'Workflow/csv/set_0.csv',
            'set_1':'Workflow/csv/set_1.csv',
            'val':'Workflow/csv/val.csv',
            'labeled':'../data/train/',
            'unlabeled':'../data/test/'
            }

    hparam = {'batch_size': 64,
            'nr_epochs': 18,
            'weight_decay': 9.428542092781991e-05,
            'dropout_rate': 0.0,
            'usize': 128,
            'penalty': 1,
            'method': ['rcrop', 'hflip'],
            'crop_ratio': 0.5,
            'crop_freq': 0.5}
    
    methods = [['hflip'], ['hflip', 'vflip'], ['hflip', 'rcrop']]
    for method in methods:
        hparam['method'] = method
        for i in range(1):
            from Workflow.Flow import Workflow
            workflow = Workflow(path=path, hparam=hparam, tag=tag)
            workflow.load_data()
            workflow.initiate_run()
            #tic = time.perf_counter()
            workflow.learn_parameters()
            #toc = time.perf_counter()
            workflow.evaluate()
            #cm = workflow.evaluator.cmatrix
            #with open('eval_test.txt', 'a') as file:
            #    file.write(f'{(cm[0,0] + cm[1,1])/cm.sum()}\t{cm[0,0]}\t{cm[0,1]}\t{cm[1,0]}\t{cm[1,1]}\t{round(toc-tic, 4)}\n')




"""
to do:
*implement mixed precision

*run as below but with resnext as convolutional block
*run same as below with one bn level in the FC block
*run the same as before with new balanced datasets
    -result:


"""










if __name__ == '__main__':
    main()