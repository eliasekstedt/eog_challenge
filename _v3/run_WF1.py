



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
            'nr_epochs': 3,
            #'architecture_name':'im',
            'weight_decay': 1e-3,
            'dropout_rate': 0.0,
            'resizes':(128, 128),
            'penalty': 1}
    
    #resizes = ['full', (50,50), (64,64), (128,128)]
    resizes = [(50,50), (64,64), (128,128)]
    for resize in resizes:
        hparam['resizes'] = resize
        print(hparam['resizes'])
        for i in range(10):
            from WF1_classifier.Flow import Workflow
            workflow = Workflow(path=path, hparam=hparam, tag=tag)
            workflow.load_data()
            workflow.initiate_run()
            workflow.learn_parameters()
            workflow.evaluate()
            cm = workflow.evaluator.cmatrix
            with open('eval_test.txt', 'a') as file:
                file.write(f'{hparam["resizes"][0]}\t{(cm[0,0] + cm[1,1])/cm.sum()}\t{cm[0,0]}\t{cm[0,1]}\t{cm[1,0]}\t{cm[1,1]}\n')


"""
remember to see what happens if cropping is done only on the lower part

and also compare to the original image reader
"""












if __name__ == '__main__':
    main()