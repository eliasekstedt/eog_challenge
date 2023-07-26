
import pandas as pd
import torch

from torchvision import transforms
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset

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

class WarwickReader(Dataset):
    def __init__(self, path_csv, path_data, flip=False, rot=False, lum=False, chan_lum=False):
        path_df = pd.read_csv(path_csv)
        self.len = len(path_df)
        self.path_images = [path_data+path_im for path_im in path_df['image']]
        self.path_labels = [path_data+path_im for path_im in path_df['label']]
        self.flip = flip
        self.rot = rot
        self.lum = lum
        self.chan_lum = chan_lum

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        #import numpy as np
        name = self.path_images[idx]
        image = read_image(self.path_images[idx])/255
        image = image.type(torch.float32)
        label = read_image(self.path_labels[idx])/255
        label = label.type(torch.long)
        # transformations
        """
        if self.flip: # horizontal, vertical or no flipping; 25, 25, 50 percent probability respectively
            factor = np.random.uniform(0,1)
            if factor < 0.25:
                image = transforms.hflip(img=image)
                label = transforms.hflip(img=label)
            elif factor > 0.75:
                image = transforms.vflip(img=image)
                label = transforms.vflip(img=label)
        if self.rot:
            if np.random.uniform(0, 1) < 0.5:
                rot_factor = 90*np.random.randint(1, 4)
                image = transforms.rotate(img=image, angle=rot_factor)
                label = transforms.rotate(img=label, angle=rot_factor)
        if self.lum:
            lum_factor = np.random.uniform(0.5, 1.5)
            image = transforms.adjust_brightness(img=image, brightness_factor=lum_factor)
        if self.chan_lum:
            lum_factor = np.random.uniform(-0.5, 0.5)
            image[0,:,:] = transforms.adjust_brightness(img=image[0,:,:], brightness_factor=1+lum_factor)
            image[1,:,:] = transforms.adjust_brightness(img=image[1,:,:], brightness_factor=1-lum_factor)
        """

        #label = label[0,:,:]
        return image, label, name