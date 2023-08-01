
import pandas as pd
import torch

from torch.utils.data import DataLoader









def main():
    path = {'traindata':'justfc/train.pt',
            'testdata':'justfc/train.pt',
            'model':'justfc/model.pth'
            }
    
    hparam = {'batch_size': 200,
              'nr_epochs':50,
              'architecture_name':'justfc',
              'weight_decay': 1e-7,
              'dropout_rate':0.0}
    
    device = 'cuda:0'

    from util.Readers import JustFCReader as Reader
    trainset = Reader(path['traindata'])
    testset = Reader(path['testdata'])
    trainloader = DataLoader(trainset, batch_size=hparam['batch_size'], shuffle=True)
    testloader = DataLoader(testset, batch_size=hparam['batch_size'], shuffle=False)

    # begin
    from util.Tools import run_init
    runpath = run_init(hparams=hparam, device=device)

    from util.Networks import JustFCNet
    model = JustFCNet(hparam['architecture_name'], hparam['weight_decay'], hparam['dropout_rate']).to(device)
    model.train_model(trainloader, testloader, hparam['nr_epochs'], runpath, device)

    # plot results
    from util.Tools import performance_plot
    performance_plot(model, runpath)









if __name__ == '__main__':
    main()