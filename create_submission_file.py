
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image



def main():
    hparam = {'batch_size': 200,
            'nr_epochs': 15,
            'architecture_name':'res18_contextual',
            'weight_decay': 1e-7,
            'dropout_rate': 0.0,
            'resizes':(128, 128)}
    
    device = 'cuda'
    runpath = 'run/29_06_59_51/'

    path_images = 'data/test/'
    csv_path = 'csv/Val.csv'

    from util.Readers import EvalContextReader
    valset = EvalContextReader(csv_path, path_images)
    valloader = DataLoader(valset, batch_size=hparam['batch_size'], shuffle=False)

    from util.Networks import ContextNet
    network = ContextNet(hparam['architecture_name'], hparam['weight_decay'], hparam['dropout_rate']).to(device)

    from util.Tools import create_submission
    create_submission(network, runpath, valloader, device)
    print('predictions generated, run finished\n')

if __name__ == '__main__':
    main()