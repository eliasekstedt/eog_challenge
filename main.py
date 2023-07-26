
# external imports
import os
from torch.utils.data import DataLoader

def main():
    # folderstructure setup
    path = {'trainmap':'csv/trainsplit.csv',
            'testmap':'csv/testsplit.csv',
            'traindata':'data/train/',
            'testdata':'data/test/'}

    device = 'cuda'

    # hyperparameters
    hparam = {'batch_size': 200,
            'nr_epochs': 10,
            'architecture_name':'standard',
            'weight_decay': 0.0,
            'dropout_rate': 0.0}

    # loading data
    from util.Readers import BasicReader
    trainset = BasicReader(path['trainmap'], path['traindata'])
    testset = BasicReader(path['testmap'], path['testdata'])
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