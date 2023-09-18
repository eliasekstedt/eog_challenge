
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
    tag = f'WF1_{str(datetime.now())[8:10]}'
    tag = tag + 'mnist_arch_assembly_test'

    path = {'set_0':'Workflow/csv/set_0.csv',
            'set_1':'Workflow/csv/set_1.csv',
            'val':'Workflow/csv/set_1.csv',
            'labeled':'../data/train/',
            'unlabeled':'../data/test/'
            }

    hparam = {'batch_size': 64,
            'nr_epochs': 1,
            'weight_decay': 9.428542092781991e-05,
            'dropout_rate': 0.0,
            'usize': 128,
            'penalty': 1,
            'method': ['hflip', 'vflip'],
            'crop_ratio': None,
            'crop_freq': None}
    

    for i in range(1):
        from Workflow.Flow import Workflow
        workflow = Workflow(path=path, hparam=hparam, tag=tag)
        workflow.load_data()
        workflow.initiate_run()
        tic = time.perf_counter()
        workflow.learn_parameters()
        toc = time.perf_counter()
        workflow.evaluate()
        cm = workflow.evaluator.cmatrix
        with open('eval_test.txt', 'a') as file:
            file.write(f'{(cm[0,0] + cm[1,1])/cm.sum()}\t{cm[0,0]}\t{cm[0,1]}\t{cm[1,0]}\t{cm[1,1]}\t{round(toc-tic, 4)}\n')




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