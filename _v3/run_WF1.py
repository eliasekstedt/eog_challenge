

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

    hparam = {'batch_size': 100,
            'nr_epochs': 18,
            'weight_decay': 1e-5,
            'dropout_rate': 0.0,
            'augment_method': None, 
            'usize': 128,
            'penalty': 1}
    
    with open('eval_test.txt', 'a') as file:
        file.write('\n#######################################')

    methods = [['ini_crop', 'lr_crop', 'hflip'], ['lr_crop', 'hflip']]#, ['hflip', 'lr_crop']]
    balances = [[128, 256]]
    for method in methods:
        hparam['augment_method'] = method
        for balance in balances:
            hparam['batch_size'] = balance[0]
            hparam['usize'] = balance[1]
            for i in range(3):
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
                    file.write(f'\n{(cm[0,0] + cm[1,1])/cm.sum()}\t{cm[0,0]}\t{cm[0,1]}\t{cm[1,0]}\t{cm[1,1]}\t{toc-tic}\t{hparam["augment_method"]}\t{hparam["batch_size"]}\t{hparam["usize"]}')















if __name__ == '__main__':
    main()