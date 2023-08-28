
import pandas as pd
import torch
from torch.utils.data import Dataset



from PIL import Image
from torchvision.transforms import RandomCrop, ToTensor, ToPILImage, Compose, RandomHorizontalFlip, Resize
#from torchvision import transforms

import matplotlib.pyplot as plt
def show(image, runpath='', title=''):
    #plt.imshow(image.to("cpu").permute(1, 2, 0))#, cmap='gray')
    plt.imshow(image.to("cpu").detach().permute(1, 2, 0))#, cmap='gray')
    #plt.title(title)
    plt.xticks([])
    plt.yticks([])
    #plt.savefig(runpath+title+'.png')
    #plt.figure()
    #plt.close('all')
    plt.show()

class Reader(Dataset):
    def __init__(self, path_csv, path_im, resizes, augment=False, eval=False):
        self.rCrop = RandomCrop(resizes[0])
        self.resize = Resize(resizes)
        self.resizes = resizes
        self.eval = eval
        self.set = pd.read_csv(path_csv)
        self.path_im = path_im
        self.augment = augment
        self.hflip = RandomHorizontalFlip()

    def __len__(self):
        return len(self.set)
    
    def __getitem__(self, idx):
        row = self.set.iloc[idx]
        filename = row['filename']
        image = Image.open(self.path_im+filename)
        width, height = image.size
        if width < self.resizes[1] or height < self.resizes[0]:
            #image = self.resize(ToTensor()(image))
            image = self.make_size_uniform(image=image, size=self.resizes)
        else:
            image = self.rCrop(ToTensor()(image))
        if self.eval:
            id = row['ID']
            return image, id, filename
        else:
            label = torch.tensor([row['extent']], dtype=torch.float32)
            if self.augment:
                image = self.hflip(image)
            return image, label, filename

    def make_size_uniform(self, image, size):
        resize = Compose([Resize(size), ToTensor()])
        image = resize(image)
        image.requires_grad_(True)
        return image
"""

from torchvision import transforms
class Reader(Dataset):
    def __init__(self, path_csv, path_im, resizes, augment=False, eval=False):
        transform = transforms.Compose([transforms.RandomHorizontalFlip()])
        self.eval = eval
        self.set = pd.read_csv(path_csv)
        self.path_im = path_im
        self.resizes = resizes
        self.augment = augment
        self.transform = transform

    def __len__(self):
        return len(self.set)
    
    def __getitem__(self, idx):
        row = self.set.iloc[idx]
        name = row['filename']
        image = Image.open(self.path_im+name)
        image = self.make_size_uniform(image=image, size=self.resizes)
        if self.eval:
            id = row['ID']
            return image, id, name
        else:
            label = torch.tensor([row['extent']], dtype=torch.float32)
            if self.augment:
                image = self.transform(image)
                show(image)
            return image, label, name

    def make_size_uniform(self, image, size):
        resize = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        image = resize(image)
        image.requires_grad_(True)
        return image

"""