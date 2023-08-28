
import pandas as pd
import torch
from torch.utils.data import Dataset



from PIL import Image
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
        filename = row['filename']
        image = Image.open(self.path_im+filename)
        image = self.make_size_uniform(image=image, size=self.resizes)
        if self.eval:
            id = row['ID']
            return image, id, filename
        else:
            label = torch.tensor([row['extent']], dtype=torch.float32)
            if self.augment:
                image = self.transform(image)
            return image, label, filename

    def make_size_uniform(self, image, size):
        resize = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        image = resize(image)
        image.requires_grad_(True)
        return image