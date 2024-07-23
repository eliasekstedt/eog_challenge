
import os

###
import sys
import numpy as np

print("Python executable:", sys.executable)
print("NumPy version:", np.__version__)
###



def main():
    ######### setable ############
    device = 'cuda:0'
    tag = 'discard'
    batch_size = 64
    nr_epochs = 2
    weight_decay = 3e-5
    state_dict = ''
    new_map = False
    eval_only = False
    ##############################
    assert not (state_dict == '' and eval_only)

    if state_dict == 'history': # if true, select most recent model
        with open('run/history/history.txt', 'r') as file:
            state_dict = file.readlines()[-1]
            while not state_dict.endswith('model.pth'):
                state_dict = state_dict[:-1]

    data_root = '../../../data/eog_224/'
    
    path = {
        'map':'csv/original/labeled.csv',
        'set_0':'csv/set_0.csv',
        'set_1':'csv/set_1.csv',
        'sample':'csv/sample.csv',
        'images': data_root,
        'state_dict':state_dict,
        }
    
    
    hparam = {
        'batch_size': batch_size,
        'nr_epochs': nr_epochs,
        'weight_decay': weight_decay,
        'use_penalty': True,
        }
    
    augmentation = {
        'center_crop': False,
        'center_block': 0, #0.1922,
        'segment_block': 0, #0.3118,
        'rotation': False,
        'hflip': True,
        'vflip': False,
        'bright': False,
    }
    
    if new_map:
        from util.splitter import Splitter
        Splitter(path_map=path['map'])

    # >--- initiate model workflow -----------------------------------------------------------------------------<
    from part.Workflow import Workflow
    workflow = Workflow(
        path=path,
        hparam=hparam,
        augmentation=augmentation,
        tag=tag,
        device=device)
    workflow.initiate_run()
    workflow.get_image_samples()
    workflow.load_data()
    workflow.load_model()
    if not eval_only:
        workflow.learn_parameters()
    workflow.evaluate()
    """
    """


if __name__ == '__main__':
    main()
