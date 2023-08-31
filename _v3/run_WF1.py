



import numpy as np
import torch
import pandas

def main():
    tag = 'initiate_WF1'

    path = {'set_0':'WF1_classifier/csv/set_0.csv',
            'set_1':'WF1_classifier/csv/set_1.csv',
            'labeled':'../data/train/',
            'unlabeled':'../data/test/'
            }

    hparam = {'batch_size': 100,
            'nr_epochs': 25,
            'weight_decay': 1e-3,
            'dropout_rate': 0.0,
            'augment_method': ['lr_crop', 'hflip'], 
            'usize': 128,
            'penalty': 1}
    
    with open('eval_test.txt', 'a') as file:
        file.write('#######################################\n')

    methods = [None, ['lr_crop', 'hflip']]
    for method in methods:
        hparam['augment_method'] = method
        for i in range(5):
            from WF1_classifier.Flow import Workflow
            workflow = Workflow(path=path, hparam=hparam, tag=tag)
            workflow.load_data()
            workflow.initiate_run()
            workflow.learn_parameters()
            workflow.evaluate()
            cm = workflow.evaluator.cmatrix
            with open('eval_test.txt', 'a') as file:
                file.write(f'{(cm[0,0] + cm[1,1])/cm.sum()}\t{cm[0,0]}\t{cm[0,1]}\t{cm[1,0]}\t{cm[1,1]}\t{hparam["usize"]}\n')


"""
remember to see what happens if cropping is done only on the lower part

and also compare to the original image reader
"""












if __name__ == '__main__':
    main()