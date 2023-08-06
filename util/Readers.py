
import pandas as pd
import torch

from torchvision import transforms
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn

def showdiff(before, after):
    import matplotlib.pyplot as plt
    before = before.detach()
    after = after.detach()
    fig, (ax1, ax2) = plt.subplots(2)
    plt.xticks([])
    plt.yticks([])
    ax1.imshow(before.to("cpu").permute(1, 2, 0))
    ax2.imshow(after.to("cpu").permute(1, 2, 0))
    #ax2.xticks([])
    #ax2.yticks([])
    plt.tight_layout()
    plt.show()

def show(image):
    import matplotlib.pyplot as plt
    plt.imshow(image.to("cpu").permute(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

class SSCropReader(Dataset):
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
        before = image.clone()
        image = self.ssCrop(image)
        after = image.clone().squeeze(0)
        showdiff(before, after)
        return image, context, label, name

    def ssCrop(self, image):
        image = image.unsqueeze(0)
        model = maskrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        model = model.cuda()
        with torch.no_grad():
            ss_results = model(image)[0]
            labels = ss_results['labels'].cuda()
            print(f'labels:\n{labels}')
            masks = ss_results['masks'].cuda()
            indices64 = torch.where(labels == 64)[0]
            indices56 = torch.where(labels == 56)[0]
            indices = torch.cat([indices64, indices56], dim=0)
            print(f'indices64:\n{indices}')
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
        #toPIL = transforms.ToPILImage()
        #image = image.squeeze(0)
        #image = toPIL(image)
        ###
        resize = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        image = resize(image)
        image.requires_grad_(True)
        return image.cuda()

class EvalSSCropReader(Dataset):
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
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
])
class Res18FCReader(Dataset):
    def __init__(self, path_csv, path_im, resizes, augment=False, eval=False):
        self.eval = eval
        self.map = pd.read_csv(path_csv).head(5300)
        self.path_im = path_im
        self.resizes = resizes
        self.augment = augment
        self.transform = transform

    def __len__(self):
        return len(self.map)
    
    def __getitem__(self, idx):
        row = self.map.iloc[idx]
        nr_primer_cols = len(self.map.columns)-16
        context = torch.tensor(self.map.iloc[idx, nr_primer_cols:].tolist(), dtype=torch.float32)
        name = row['filename']
        image = Image.open(self.path_im+name)
        image = self.make_size_uniform(image=image, size=self.resizes)
        if self.eval:
            id = row['ID']
            return image, context, id, name
        else:
            label = torch.tensor([row['extent']], dtype=torch.float32)
            if self.augment:
                image = transform(image)
            return image, context, label, name

    def make_size_uniform(self, image, size):
        resize = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        image = resize(image)
        image.requires_grad_(True)
        return image

### current ###
"""
class EvalRes18FCReader(Dataset):
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
        #image = image[:,image.shape[1]//2:, :]
        return image, context, id, name

    def make_size_uniform(self, image, size):
        resize = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        image = resize(image)
        image.requires_grad_(True)
        return image
"""

