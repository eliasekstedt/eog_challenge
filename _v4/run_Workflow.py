
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

def main():
    #tag = f'test_cvg'#{str(datetime.now())[8:10]}'
    #tag = tag + ''

    path = {'set_0':'Workflow/csv/set_0.csv',
            'set_1':'Workflow/csv/set_1.csv',
            'val':'Workflow/csv/val.csv',
            'labeled':'../data/train/',
            'unlabeled':'../data/test/'
            }

    hparam = {'fcv': None,
            'mode': None,
            'batch_size': 64,
            'nr_epochs': 35,
            'weight_decay': 0,#9.428542092781991e-05,
            'dropout_rate': 0.0,
            'usize': 128,
            'penalty': 1,
            'method': ['hflip', 'rcrop'],
            'crop_ratio': 0.5,
            'crop_freq': 0.5}
    
    #for key in hparam.keys():
    #    tag += f'_{hparam[key]}'


    
    modes = ['res34']
    fc_versions = ['wo3', 'bn3', 'wo2', 'bn2', 'wo1', 'bn1']
    for mode in modes:
        for fcv in fc_versions:
            hparam['fcv'], hparam['mode'] = fcv, mode
            tag = f'cvg_{fcv}_{mode}'
            from Workflow.Flow import Workflow
            workflow = Workflow(path=path, hparam=hparam, tag=tag)
            workflow.load_data()
            workflow.initiate_run()
            workflow.learn_parameters()
            workflow.evaluate()




"""
to do:
*implement mixed precision
* run with and without bn, no weight decay, no dropout. both res34 and mobv2. hoping
to see a degree of overfitting, then slowly improve from there.
    -result:

*optimize dropout with weight decay, then dropout detailed
    -result: weight decay seems too high at ~1e-4. clear underfitting always.
    allow a degree of overfitting and let other regularization techniques do
    their part.
*run as below but with resnext as convolutional block
    -result: bout the same
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