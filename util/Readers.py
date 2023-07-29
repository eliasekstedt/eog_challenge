
import pandas as pd
import torch

from torchvision import transforms
from PIL import Image
#from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn

def show(image, runpath='', title=''):
    import matplotlib.pyplot as plt
    image = image.detach()
    plt.imshow(image.to("cpu").permute(1, 2, 0))#, cmap='gray')
    #plt.title(title)
    plt.xticks([])
    plt.yticks([])
    #plt.savefig(runpath+title+'.png')
    #plt.figure()
    #plt.close('all')
    plt.show()


class SSCropCxtReader(Dataset):
    def __init__(self, path_csv, path_im):
        self.path_im = path_im
        self.map = pd.read_csv(path_csv)

    def __len__(self):
        return len(self.map)
    
    def __getitem__(self, idx):
        row = self.map.iloc[idx]
        context = torch.tensor(self.map.iloc[idx, 3:].tolist(), dtype=torch.float32)
        label = torch.tensor([row['extent']], dtype=torch.float32)
        name = row['filename']
        image = Image.open(self.path_im+name)
        image = self.ssCrop(image)
        image = self.make_size_uniform(image=image, size=(128, 128))
        return image, context, label, name

    def ssCrop(self, image):
        totensor = transforms.ToTensor()
        image = totensor(image).cuda()
        image = image.unsqueeze(0)
        model = maskrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        model = model.cuda()
        with torch.no_grad():
            print(image.shape)
            ss_results = model(image)[0]
            labels = ss_results['labels'].cuda()
            print(f'labels:\n{labels}')
            masks = ss_results['masks'].cuda()
            indices = torch.where(labels == 64)[0]
            masks = masks[indices]
            masks = torch.where(masks > 0.5, torch.tensor(1.0), torch.tensor(0.0))
            masks = masks.sum(dim=0)
            masks = torch.where(masks > 0, torch.tensor([1]).cuda(), torch.tensor([0]).cuda())
            masks = masks.expand_as(image)
            masks = (masks == 1)
            image = image*masks
        return image

    def make_size_uniform(self, image, size):
        ###
        toPIL = transforms.ToPILImage()
        image = image.squeeze(0)
        image = toPIL(image)
        ###
        resize = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        image = resize(image)
        image.requires_grad_(True)
        return image

class EvalSSCropCxtReader(Dataset):
    def __init__(self, path_csv, path_im, resizes):
        self.resizes = resizes
        self.path_im = path_im
        self.map = pd.read_csv(path_csv)

    def __len__(self):
        return len(self.map)
    
    def __getitem__(self, idx):
        row = self.map.iloc[idx]
        context = torch.tensor(self.map.iloc[idx, 2:].tolist(), dtype=torch.float32)
        id = row['ID']
        name = row['filename']
        image = Image.open(self.path_im+name)
        image = self.make_size_uniform(image=image, size=self.resizes)
        return image, context, id, name

    def make_size_uniform(self, image, size):
        resize = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        image = resize(image)
        image.requires_grad_(True)
        return image

### current ###
class ContextReader(Dataset):
    def __init__(self, path_csv, path_im, resizes):
        self.resizes = resizes
        self.path_im = path_im
        self.map = pd.read_csv(path_csv)

    def __len__(self):
        return len(self.map)
    
    def __getitem__(self, idx):
        row = self.map.iloc[idx]
        context = torch.tensor(self.map.iloc[idx, 3:].tolist(), dtype=torch.float32)
        label = torch.tensor([row['extent']], dtype=torch.float32)
        name = row['filename']
        image = Image.open(self.path_im+name)
        image = self.make_size_uniform(image=image, size=self.resizes)
        # skycrop
        image = image[:,image.shape[1]//2:, :]
        return image, context, label, name

    def make_size_uniform(self, image, size):
        resize = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        image = resize(image)
        image.requires_grad_(True)
        return image

class EvalContextReader(Dataset):
    def __init__(self, path_csv, path_im, resizes):
        self.resizes = resizes
        self.path_im = path_im
        self.map = pd.read_csv(path_csv)

    def __len__(self):
        return len(self.map)
    
    def __getitem__(self, idx):
        row = self.map.iloc[idx]
        context = torch.tensor(self.map.iloc[idx, 2:].tolist(), dtype=torch.float32)
        id = row['ID']
        name = row['filename']
        image = Image.open(self.path_im+name)
        image = self.make_size_uniform(image=image, size=self.resizes)
        # skycrop
        image = image[:,image.shape[1]//2:, :]
        return image, context, id, name

    def make_size_uniform(self, image, size):
        resize = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        image = resize(image)
        image.requires_grad_(True)
        return image





class BasicReader(Dataset):
    def __init__(self, path_csv, path_im):
        self.path_im = path_im
        self.map = pd.read_csv(path_csv)

    def __len__(self):
        return len(self.map)
    
    def __getitem__(self, idx):
        row = self.map.iloc[idx]
        label = torch.tensor([row['extent']], dtype=torch.float32)
        name = row['filename']
        image = Image.open(self.path_im+name)
        image = self.make_size_uniform(image=image, size=(128, 128))
        return image, label, name

    def make_size_uniform(self, image, size):
        resize = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        image = resize(image)
        image.requires_grad_(True)
        return image

class EvalBasicReader(Dataset):
    def __init__(self, path_csv, path_im):
        self.path_im = path_im
        self.map = pd.read_csv(path_csv)

    def __len__(self):
        return len(self.map)
    
    def __getitem__(self, idx):
        row = self.map.iloc[idx]
        
        id = row['ID']
        name = row['filename']
        image = Image.open(self.path_im+name)
        image = self.make_size_uniform(image=image, size=(128, 128))
        return image, id, name

    def make_size_uniform(self, image, size):
        resize = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        image = resize(image)
        image.requires_grad_(True)
        return image

