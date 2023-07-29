
# external imports
import os
from torch.utils.data import DataLoader

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main():
    # folderstructure setup
    path = {'trainmap':'csv/trainsplit.csv',
            'testmap':'csv/testsplit.csv',
            'valmap':'csv/Val.csv',
            'data_labeled':'data/train/',
            'data_unlabeled':'data/test/'
            }

    device = 'cuda:0'

    # hyperparameters
    hparam = {'batch_size': 200,
            'nr_epochs': 15,
            'architecture_name':'res18_contextual',
            'weight_decay': 1e-7,
            'dropout_rate': 0.0,
            'resizes':(256, 128)}

    # loading data
    from util.Readers import ContextReader as Reader
    trainset = Reader(path['trainmap'], path['data_labeled'], resizes=hparam['resizes'])
    testset = Reader(path['testmap'], path['data_labeled'], resizes=hparam['resizes'])
    trainloader = DataLoader(trainset, batch_size=hparam['batch_size'], shuffle=True)
    testloader = DataLoader(testset, batch_size=hparam['batch_size'], shuffle=True)

    # begin
    from util.Tools import run_init
    runpath = run_init(hparams=hparam, device=device)

    from util.Networks import ContextNet
    model = ContextNet(hparam['architecture_name'], hparam['weight_decay'], hparam['dropout_rate']).to(device)
    model.train_model(trainloader, testloader, hparam['nr_epochs'], runpath, device)

    # plot results
    from util.Tools import performance_plot
    performance_plot(model, runpath)

    # generating submission file
    from util.Readers import EvalContextReader
    valset = EvalContextReader(path['valmap'], path['data_unlabeled'], resizes=hparam['resizes'])
    valloader = DataLoader(valset, batch_size=hparam['batch_size'], shuffle=False)
    network = ContextNet(hparam['architecture_name'], hparam['weight_decay'], hparam['dropout_rate']).to(device)

    from util.Tools import create_submission
    create_submission(network, runpath, valloader, device)
    print('predictions generated, run finished\n')


if __name__ == '__main__':
    main()