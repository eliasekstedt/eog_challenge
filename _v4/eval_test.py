
import torch
from torch.utils.data import DataLoader

from Workflow.parts.Evaluation import Evaluator
from Workflow.parts.Evaluation import Heatmap
from Workflow.parts.Networks import Net
from Workflow.parts.Readers import Reader

device = 'cuda:0'

path = {'set_0':'Workflow/csv/set_0.csv',
        'set_1':'Workflow/csv/set_1.csv',
        'val':'Workflow/csv/val.csv',
        'labeled':'../data/train/',
        'unlabeled':'../data/test/'}

hparam = {'batch_size': 64,
            'nr_epochs': 3,
            'weight_decay': 9.428542092781991e-05,
            'dropout_rate': 0.0,
            'usize': 128,
            'penalty': 1,
            'method': ['hflip', 'vflip'],
            'crop_ratio': None,
            'crop_freq': None}


runpath = 'run/WF1_18_v4_initial/18_13_55_23/'
model = Net(hparam['weight_decay'], hparam['dropout_rate'], hparam['penalty']).to(device)
model.load_state_dict(torch.load(f'{runpath}model.pth'))
set_1 = Reader(path['set_1'], path['labeled'], usize=hparam['usize'], augment_method=[], crop_ratio=None, crop_freq=None, eval=eval)
loader_1 = DataLoader(set_1, batch_size=hparam['batch_size'], shuffle=False)
evaluator = Evaluator(runpath, model, loader_1, path['set_1'], device)
heatmap = Heatmap(runpath, height=220, vlines=True)
heatmap.save(runpath)