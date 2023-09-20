
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
    tag = tag + 'mobV2'

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
            'method': ['hflip', 'rcrop'],
            'crop_ratio': 0.5,
            'crop_freq': 0.5}
    
    
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
    -result:
*without damage
    -result: worse; did not converge as quickly and probably not as far had i 
    let it go all the way.
*same as before but shorter and wider fc component
    -result: slightly worse
*run same as below with one bn level in the FC block
    -result: bout the same
*run the same as before with new balanced datasets
    -result: [hflip, rcrop], unlike the others never got to the point of overfitting.
    also slightly lower min than the others. the augmentations i will go with for now.
    dont know why i did not get that result as clearly before.


"""










if __name__ == '__main__':
    main()