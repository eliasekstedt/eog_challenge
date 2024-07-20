
import os

def main():
    ######### setable ############
    device = 'cuda:0'
    tag = 'discard'
    name_dataset = 'eog'
    batch_size = 64
    nr_epochs = 120
    weight_decay = 3e-5
    state_dict = ''
    force_new_map = False
    eval_only = False
    ##############################
    assert not (state_dict == '' and eval_only)

    if state_dict == 'history': # if true, select most recent model
        with open('run/history/history.txt', 'r') as file:
            state_dict = file.readlines()[-1]
            while not state_dict.endswith('model.pth'):
                state_dict = state_dict[:-1]

    with open('../../../data/path_to_data.txt') as file:
        data_root = file.readline()[:-1]
    data_root = '../../../data/eog/'
    path_map = { # for creating data maps as csv files for model workflow
        'data_root': f"{data_root}{name_dataset}/",
        'dir_class_0':'009/',
        'dir_class_1':'025/',
        'dir_class_2':'053/',
        'dir_class_3':'063/',
        'dir_class_4':'067/',
        'dir_class_5':'071/',
        'dir_class_6':'077/',
        'dir_class_7':'081/',
        'dir_class_8':'086/',
        'dir_class_9':'096/',
        'map':f"csv/original/{name_dataset}.csv",
    }
    
    path_model = {
        'set_0':'csv/set_0.csv',
        'set_1':'csv/set_1.csv',
        'sample':'csv/sample.csv',
        'images': path_map['data_root'],
        'state_dict':state_dict,
        }
    
    hparam = {
        'batch_size': batch_size,
        'nr_epochs': nr_epochs,
        'weight_decay': weight_decay,
        }
    
    augmentation = {
        #'image_size': image_size,
        'center_crop': False,
        'center_block': 0.1922,
        'segment_block': 0.3118,
        'rotation': True,
        'hflip': True,
        'vflip': False,
        'bright': False,
    }

    # handle .csv maps for data
    if not os.path.exists(path_map['map']) or force_new_map:
        from data_tools.create_csv import MapWriter, Splitter
        MapWriter(path_map)
        max = None
        Splitter(path_map, max)
    
    # initiate model workflow
    from part.Workflow import Workflow
    workflow = Workflow(
        path=path_model,
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
