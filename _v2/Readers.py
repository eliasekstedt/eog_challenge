
import pandas as pd
import torch
from torch.utils.data import Dataset


class FCReader(Dataset):
    def __init__(self, path_csv, eval=False):
        self.eval = eval
        self.data = pd.read_csv(path_csv)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        nr_primer_cols = len(self.data.columns)-16
        context = torch.tensor(self.data.iloc[idx, nr_primer_cols:].tolist(), dtype=torch.float32)
        if self.eval:
            id = row['ID']
            return context, id
        else:
            label = torch.tensor([row['extent']], dtype=torch.float32)
            return context, label

from PIL import Image
from torchvision import transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
])
class ImReader(Dataset):
    def __init__(self, path_csv, path_im, resizes, augment=False, eval=False):
        self.eval = eval
        self.map = pd.read_csv(path_csv)
        self.path_im = path_im
        self.resizes = resizes
        self.augment = augment
        self.transform = transform

    def __len__(self):
        return len(self.map)
    
    def __getitem__(self, idx):
        row = self.map.iloc[idx]
        name = row['filename']
        image = Image.open(self.path_im+name)
        image = self.make_size_uniform(image=image, size=self.resizes)
        if self.eval:
            id = row['ID']
            return image, id, name
        else:
            label = torch.tensor([row['extent']], dtype=torch.float32)
            if self.augment:
                image = transform(image)
            return image, label, name

    def make_size_uniform(self, image, size):
        resize = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        image = resize(image)
        image.requires_grad_(True)
        return image