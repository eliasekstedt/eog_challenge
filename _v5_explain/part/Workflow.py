
import timm
import torch.nn as nn
from torch.utils.data import DataLoader
from util.tools_filing import file_it
import pandas as pd


class Workflow:
    def __init__(self, path, hparam, augmentation, tag, device):
        self.device = device
        self.tag = tag
        self.path = path
        self.hparam = hparam
        self.augmentation = augmentation
        self.runpath = None
        self.loader_0 = None
        self.loader_1 = None
        self.protocol = None
        self.evaluator = None

    def get_image_samples(self):
        print('creating samples')
        import torch
        from part.Reader import Reader
        sampleset = Reader(
            self.path['sample'],
            self.path['images'],
            self.augmentation,
            mode='sample',
            )
        loader = DataLoader(sampleset, batch_size=1, shuffle=False)
        images_ori, images_aug = [], []
        for ori, aug in loader:
            images_ori.append(ori.squeeze(0))
            images_aug.append(aug.squeeze(0))
        from util.tools_plotting import make_tensor_assembly
        assembly_ori = make_tensor_assembly(images_ori)
        assembly_aug = make_tensor_assembly(images_aug)
        divider = torch.zeros_like(assembly_ori)[:,:, :10]
        full_assembly = torch.cat([assembly_ori, divider, assembly_aug], dim=2)
        import torchvision.transforms as transforms
        #from PIL import Image
        to_pil = transforms.ToPILImage()
        full_assembly = to_pil(full_assembly)
        full_assembly.save(f"{self.runpath}sample.png")

    def load_data(self):
        self.loader_0 = self.get_loader(
            path_csv=self.path['set_0'],
            path_im=self.path['images'],
            mode = 'train',
            batch_size=self.hparam['batch_size'],
            shuffle=True,
            )
        self.loader_1 = self.get_loader(
            path_csv=self.path['set_1'],
            path_im=self.path['images'],
            mode = 'train',
            batch_size=self.hparam['batch_size'],
            shuffle=True,
            )
        self.evalloader = self.get_loader( # evaluation data, same as testing data
            path_csv=self.path['set_1'],
            path_im=self.path['images'],
            mode = 'eval',
            batch_size=self.hparam['batch_size'],
            shuffle=False,
            )

    def get_loader(self, path_csv, path_im, mode, batch_size, shuffle):
        from part.Reader import Reader
        dataset = Reader(path_csv, path_im, self.augmentation, mode)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def initiate_run(self):
        from util.tools_filing import run_init
        self.runpath = run_init(self.hparam, self.augmentation, self.tag, self.device)
    
    def load_model(self):
        def get_penalties(path_set_0):
            set_0 = pd.read_csv(path_set_0)
            counts = set_0['label'].value_counts()
            counts = counts.tolist()
            counts = [1 - i / sum(counts) for i in counts]
            penalty = torch.tensor([i / min(counts) for i in counts])
            return penalty

        import torch
        from part.Model import CMOS31
        model = CMOS31()
        if not self.path['state_dict'] == '':
            model.load_state_dict(torch.load(self.path['state_dict'], map_location='cuda:0'))
            message = f"loaded state: {self.path['state_dict']}\n"
            file_it(f'{self.runpath}log.txt', message, True)
        model.to(self.device)
        from part.Protocol import Protocol
        if self.hparam['use_penalty']:
            penalty = get_penalties(self.path['set_0'])
        else:
            penalty = None
        print(f"penalty: {penalty}")
        self.protocol = Protocol(self.hparam, model, penalty.to(self.device))

    def learn_parameters(self):
        self.protocol.execute_protocol(self.loader_0, self.loader_1, self.hparam['nr_epochs'], self.runpath, self.device)

    def evaluate(self):
        print('evaluating...')
        from util.tools_plotting import plot_performance
        plot_performance(self.protocol, self.runpath)
        from part.Evaluator import Evaluator
        self.evaluator = Evaluator(self.runpath, self.path['set_1'], self.protocol.model, self.evalloader, self.device)
        print('...complete')