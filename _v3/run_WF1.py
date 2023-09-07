

from datetime import datetime
import time
import numpy as np
import torch
import pandas

def main():
    tag = f'WF1_{str(datetime.now())[8:10]}'

    path = {'set_0':'WF1_classifier/csv/set_0.csv',
            'set_1':'WF1_classifier/csv/set_1.csv',
            'labeled':'../data/train/',
            'unlabeled':'../data/test/'
            }

    hparam = {'batch_size': 64,
            'nr_epochs': 20,
            'weight_decay': 1e-5,
            'dropout_rate': 0.0,
            'augment_method': ['rcrop', 'hflip'],
            'crop_ratio': None,
            'usize': 128,
            'penalty': 1}
    
    with open('eval_test.txt', 'a') as file:
        file.write('\n#######################################')

    crop_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # again restricted to lower for comparison
    for ratio in crop_ratios:
        hparam['crop_ratio'] = ratio
        for i in range(2):
            from WF1_classifier.Flow import Workflow
            workflow = Workflow(path=path, hparam=hparam, tag=tag)
            workflow.load_data()
            workflow.initiate_run()
            tic = time.perf_counter()
            workflow.learn_parameters()
            toc = time.perf_counter()
            workflow.evaluate()
            cm = workflow.evaluator.cmatrix
            with open('eval_test.txt', 'a') as file:
                file.write(f'\n{(cm[0,0] + cm[1,1])/cm.sum()}\t{cm[0,0]}\t{cm[0,1]}\t{cm[1,0]}\t{cm[1,1]}\t{toc-tic}\t{hparam["crop_ratio"]}')




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