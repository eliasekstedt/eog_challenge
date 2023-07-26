
# external imports
import os
from torch.utils.data import DataLoader

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main():
    # folderstructure setup
    path = {'trainmap':'csv/trainsplit.csv',
            'testmap':'csv/testsplit.csv',
            'data_labeled':'data/train/',
            'data_unlabeled':'data/test/'
            }

    device = 'cuda'

    # hyperparameters
    hparam = {'batch_size': 200,
        'nr_epochs': 10,
        'architecture_name':'res18_unintegrated',
        'weight_decay': 0.0,
        'dropout_rate': 0.0}

    # loading data
    from util.Readers import BasicReader as Reader
    trainset = Reader(path['trainmap'], path['data_labeled'])
    testset = Reader(path['testmap'], path['data_labeled'])
    trainloader = DataLoader(trainset, batch_size=hparam['batch_size'], shuffle=True)
    testloader = DataLoader(testset, batch_size=hparam['batch_size'], shuffle=True)

    # begin
    from util.Tools import run_init
    runpath = run_init(hparams=hparam, device=device)

    from util.Networks import Orinet
    model = Orinet(hparam['architecture_name'], hparam['weight_decay'], hparam['dropout_rate']).to(device)
    model.train_model(trainloader, testloader, hparam['nr_epochs'], runpath, device)




if __name__ == '__main__':
    main()