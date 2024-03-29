
# external imports
import os
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip

# make deterministic
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from skopt import gp_minimize
from skopt.space import Real

def objective(param):
    # folderstructure setup
    path = {'fold_0':'csv/fold_0.csv',
            'fold_1':'csv/fold_1.csv',
            'valmap':'csv/Val.csv',
            'data_labeled':'data/train/',
            'data_unlabeled':'data/test/'
            }

    tag = 'gp_crossval'
    device = 'cuda:0'

    # hyperparameters
    hparam = {'batch_size': 100,
            'nr_epochs': 25,
            'architecture_name':'res18fc',
            'weight_decay': param[0],
            'dropout_rate': 0.0,
            'resizes':(128, 128),
            'penalty': 1}

    # loading data
    from util.Readers import Res18FCReader as Reader
    set0 = Reader(path['fold_0'], path['data_labeled'], resizes=hparam['resizes'], augment=True)
    loader0 = DataLoader(set0, batch_size=hparam['batch_size'], shuffle=True)
    set1 = Reader(path['fold_1'], path['data_labeled'], resizes=hparam['resizes'], augment=True)
    loader1 = DataLoader(set1, batch_size=hparam['batch_size'], shuffle=True)

    # begin
    from util.Tools import run_init
    runpath = run_init(hparams=hparam, tag=tag, device=device)

    from util.Networks import Res18FCNet
    from util.Tools import performance_plot
    model0 = Res18FCNet(0, hparam['architecture_name'], hparam['weight_decay'], hparam['dropout_rate'], hparam['penalty']).to(device)
    model0.train_model(trainloader=loader0, testloader=loader1, nr_epochs=hparam['nr_epochs'], runpath=runpath, device=device)
    performance_plot(model0, runpath)
    model1 = Res18FCNet(1, hparam['architecture_name'], hparam['weight_decay'], hparam['dropout_rate'], hparam['penalty']).to(device)
    model1.train_model(trainloader=loader1, testloader=loader0, nr_epochs=hparam['nr_epochs'], runpath=runpath, device=device)
    performance_plot(model1, runpath)

    # evaluation
    from util.Networks import Res18FCNet as Network

    # creating and loading model state here because could not be passed as a parameter for some reason
    models = [Network(fold=None, architecture_name=hparam['architecture_name'], weight_decay=hparam['weight_decay'], dropout_rate=hparam['dropout_rate'], penalty=hparam['penalty']).to(device), Network(fold=None, architecture_name=hparam['architecture_name'], weight_decay=hparam['weight_decay'], dropout_rate=hparam['dropout_rate'], penalty=hparam['penalty']).to(device)]
    for i, model in enumerate(models):
        model.load_state_dict(torch.load(runpath+'model_'+str(i)+'.pth'))
        model.eval()

    from util.Readers import Res18FCReader as Reader
    from util.Evaluators import CrossEvaluator as Evaluator
    evaluator = Evaluator(runpath, models, Reader, path, hparam, device)
    print(f'evaldata\n{evaluator.evaldata}')
    print(f'shape: {evaluator.evaldata.shape}')
    print(f'param|rmse: {param}|{evaluator.rmse}')
    evaluator.evaldata.to_csv(f'{runpath}eval_data_{evaluator.rmse}.csv', index=False)
    return float(evaluator.rmse)

def main():
    # Define hyperparameter space
    space = [Real(0, 1e-6, name='wd')]
    result = gp_minimize(objective, space, n_calls=10, acq_func='EI', n_random_starts=5)
    print('optimization finished\n')
    best_param = result.x[0]
    print(result)
    print(f'best_param: {best_param}')

    


if __name__ == '__main__':
    main()