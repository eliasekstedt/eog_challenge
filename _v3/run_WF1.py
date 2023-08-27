





def main():
    tag = 'initiate_WF1'

    path = {'set_0':'WF1_classifier/csv/set_0.csv',
            'set_1':'WF1_classifier/csv/set_1.csv',
            'labeled':'../data/train/',
            'unlabeled':'../data/test/'
            }

    hparam = {'batch_size': 100,
            'nr_epochs': 9,
            'architecture_name':'im',
            'weight_decay': 1e-3,
            'dropout_rate': 0.0,
            'resizes':(128, 128),
            'penalty': 1}
    
    from WF1_classifier.Flow import Workflow
    workflow = Workflow(path=path, hparam=hparam, tag=tag)
    workflow.load_data()
    workflow.initiate_run()
    workflow.learn_parameters()
    workflow.evaluate()
    #workflow.get_heatmap()















if __name__ == '__main__':
    main()