





def main():
    tag = 'initiate_wf_0'

    path = {'set_0':'csv/sub_0.csv',
            'set_1':'csv/sub_1.csv',
            #'set_v':'csv/sub_v.csv',
            'labeled':'../data/train/',
            'unlabeled':'../data/test/'
            }

    hparam = {'batch_size': 100,
            'nr_epochs': 3,
            'architecture_name':'im',
            'weight_decay': 1e-3,
            'dropout_rate': 0.0,
            'resizes':(128, 128),
            'penalty': 1}
    
    from Workflow_0.Flow import Workflow
    workflow = Workflow(path=path, hparam=hparam, tag=tag)
    workflow.load_data()
    workflow.initiate_run()
    workflow.learn_parameters()
    workflow.evaluate()
    workflow.get_heatmap()















if __name__ == '__main__':
    main()