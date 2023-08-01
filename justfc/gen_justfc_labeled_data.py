
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader



def tsave(image, runpath='', title=''):
    print(image.shape)
    plt.imshow(image.to("cpu").permute(1, 2, 0))#, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'justfc/{title}.jpg')
    plt.figure()
    plt.close('all')
    #plt.show()

class JustFCPrinter:
    def __init__(self, path_csv, path_im, resizes, network):
        self.network = network
        self.resizes = resizes
        self.path_im = path_im
        self.map = pd.read_csv(path_csv)

    def __len__(self):
        return len(self.map)
    
    def __getitem__(self, idx):
        row = self.map.iloc[idx]
        context = torch.tensor(self.map.iloc[idx, 3:].tolist(), dtype=torch.float32, device='cuda:0').unsqueeze(0)
        label = torch.tensor([row['extent']], dtype=torch.float32, device='cuda:0')
        name = row['filename']
        image = Image.open(self.path_im+name)
        image = self.make_size_uniform(image=image, size=self.resizes)
        image = image.unsqueeze(0)
        pred, rescomponent = self.network(image.cuda(), context)
        return rescomponent.to('cuda:0'), context, label, pred.to('cuda:0'), name[0]
    
    def make_size_uniform(self, image, size):
        resize = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        image = resize(image)
        image.requires_grad_(True)
        return image.to('cuda:0')


def gen_tensor_data(loader, title):
    justfc_data = None
    names = None
    labels = None
    preds = None
    with torch.no_grad():
        for i, (rescomponent, context, label, pred, name) in enumerate(loader):
            rescomponent = rescomponent.squeeze(0)
            context = context.squeeze(0)
            pred = pred.squeeze(0)
            component_assembly = torch.cat([rescomponent, context, label, pred], dim=1)
            if justfc_data is None:
                justfc_data = component_assembly
                names = [name]
                labels = [label.item()]
                preds = [pred.item()]
            else:
                justfc_data = torch.cat([justfc_data, component_assembly], dim=0)
                names = names + [name]
                labels = labels + [label.item()]
                preds = preds + [pred.item()]

            if i%500 == 0 and i != 0:
                print(justfc_data.shape)

    df = pd.DataFrame({'name':names, 'label':labels, 'pred':preds})
    df.to_csv(f'justfc/df_{title}.csv', index=False)
    torch.save(justfc_data, f'justfc/{title}.pt')


def main():
    path = {'trainmap':'csv/trainsplit.csv',
            'testmap':'csv/testsplit.csv',
            'valmap':'csv/Val.csv',
            'data_labeled':'data/train/',
            'data_unlabeled':'data/test/'
            }
    
    hparam = {'batch_size': 200,
            'nr_epochs': 15,
            'architecture_name':'justres',
            'weight_decay': 1e-7,
            'dropout_rate': 0.0,
            'resizes':(128, 128)}
    
    device = 'cuda:0'

    train_network = False
    if train_network:
        # loading data
        from util.Readers import ContextReader as Reader
        trainset = Reader(path['trainmap'], path['data_labeled'], resizes=hparam['resizes'])
        testset = Reader(path['testmap'], path['data_labeled'], resizes=hparam['resizes'])
        trainloader = DataLoader(trainset, batch_size=hparam['batch_size'], shuffle=True)
        testloader = DataLoader(testset, batch_size=hparam['batch_size'], shuffle=False)

        # begin
        from util.Tools import run_init
        runpath = run_init(hparams=hparam, device=device)

        from util.Networks import JustResNet
        model = JustResNet(hparam['architecture_name'], hparam['weight_decay'], hparam['dropout_rate']).to(device)
        model.train_model(trainloader, testloader, hparam['nr_epochs'], runpath, device)
    
    create_files = True
    if create_files:
        if train_network is False:
            runpath = 'justfc/'
        # loading pretrained model which outputs rescomponent for each image (along with labels and other features)
        from util.Networks import JustResNet
        network = JustResNet(hparam['architecture_name'], hparam['weight_decay'], hparam['dropout_rate']).to(device)
        network.load_state_dict(torch.load(runpath+'model.pth'))
        network.eval()

        # dataloaders for fc input data
        trainset = JustFCPrinter(path['trainmap'], path['data_labeled'], resizes=hparam['resizes'], network=network)
        testset = JustFCPrinter(path['testmap'], path['data_labeled'], resizes=hparam['resizes'], network=network)
        trainloader = DataLoader(trainset, batch_size=1, shuffle=False)
        testloader = DataLoader(testset, batch_size=1, shuffle=False)
        
        # fc input data to file
        gen_tensor_data(trainloader, 'train')
        gen_tensor_data(testloader, 'test')
    

if __name__ == '__main__':
    main()








# Load pre-trained model
#resnet18 = models.resnet18(pretrained=True)
# Remove the last layer (usually a softmax or similar layer)
#esnet18 = nn.Sequential(*list(resnet18.children())[:-1])


"""
from util.Networks import ContextNet
network = ContextNet('res18_contextual', 1e-7, 0.0).to(device)
network.load_state_dict(torch.load('justfc/jfcmodel.pth'))
ch = list(network.children())
print(ch[1][-1])
#network = nn.Sequential(*list(network.children())[:-1])
network.eval()
"""