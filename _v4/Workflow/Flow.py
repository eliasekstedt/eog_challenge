
import torch
from torch.utils.data import DataLoader

# part imports
from Workflow.parts.Readers import Reader
from Workflow.parts.Tools import run_init
from Workflow.parts.Tools import plot_performance
from Workflow.parts.Tools import plot_by_ctx_feature
from Workflow.parts.Networks import Net
from Workflow.parts.Evaluation import Evaluator
from Workflow.parts.Evaluation import Heatmap


class Workflow:
    def __init__(self, path, hparam, tag='unspecified'):
        self.device = 'cuda:0'
        self.tag = tag
        self.path = path
        self.hparam = hparam
        self.runpath = None
        self.loader_0 = None
        self.loader_1 = None
        self.evalloader = None
        self.model = None
        self.evaluator = None
        self.heatmap = None
        
    def load_data(self):
        self.loader_0 = self.get_loader(self.path['set_0'], self.path['labeled'], augment_method=self.hparam['method'], crop_ratio=self.hparam['crop_ratio'], crop_freq=self.hparam['crop_freq'], eval=False, batch_size=self.hparam['batch_size'], shuffle=True)
        self.loader_1 = self.get_loader(self.path['set_1'], self.path['labeled'], augment_method=[], crop_ratio=None, crop_freq=None, eval=False, batch_size=self.hparam['batch_size'], shuffle=True)
        self.evalloader = self.get_loader(self.path['set_1'], self.path['labeled'], augment_method=[], crop_ratio=None, crop_freq=None, eval=True, batch_size=self.hparam['batch_size'], shuffle=False)

    def get_loader(self, path_csv, path_im, augment_method, crop_ratio, crop_freq, eval, batch_size, shuffle):
        set = Reader(path_csv, path_im, usize=self.hparam['usize'], augment_method=augment_method, crop_ratio=crop_ratio, crop_freq=crop_freq, eval=eval)
        return DataLoader(set, batch_size=batch_size, shuffle=shuffle)
    
    def initiate_run(self):
        self.runpath = run_init(hparams=self.hparam, tag=self.tag, device=self.device)

    def learn_parameters(self):
        self.model = Net(self.hparam['weight_decay'], self.hparam['dropout_rate'], self.hparam['penalty'], self.hparam['mode']).to(self.device)
        self.model.train_model(self.loader_0, self.loader_1, self.hparam['nr_epochs'], self.runpath, self.device)

    def evaluate(self):
        print('evaluating')
        plot_performance(self.model, self.runpath)
        self.model.load_state_dict(torch.load(f'{self.runpath}sp2_model.pth'))
        self.evaluator = Evaluator(self.runpath, self.model, self.evalloader, self.path['set_1'], self.device)
        heatmap = Heatmap(self.runpath, height=220, vlines=True)
        heatmap.save(self.runpath)





