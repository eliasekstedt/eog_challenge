
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


from PIL import Image
from torchvision.transforms import RandomCrop, ToTensor, ToPILImage, Compose, RandomHorizontalFlip, Resize
from torch.nn.functional import interpolate
#from torchvision import transforms

import matplotlib.pyplot as plt
def show(image, runpath='', title=''):
    #plt.imshow(image.to("cpu").permute(1, 2, 0))#, cmap='gray')
    plt.imshow(image.to("cpu").detach().permute(1, 2, 0))#, cmap='gray')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    #plt.savefig(runpath+title+'.png')
    #plt.figure()
    #plt.close('all')
    plt.show()

class Reader(Dataset):
    def __init__(self, path_csv, path_im, usize, augment_method=[], eval=False):
        self.blocker = Fblocker()
        self.augment_method = augment_method
        self.usize = usize
        self.set = pd.read_csv(path_csv)
        self.path_im = path_im
        self.eval = eval

    def __len__(self):
        return len(self.set)
    
    def __getitem__(self, idx):
        row = self.set.iloc[idx]
        filename = row['filename']
        image = read_image(f'{self.path_im}{filename}')/255
        image = image.type(torch.float32)
        #title = f'{row["extent"]}| {row["growth_stage_F"]}, {row["growth_stage_M"]}, {row["growth_stage_S"]}, {row["growth_stage_V"]}'
        #show(image, title=title)
        image = self.augment(image)
        if self.eval:
            id = row['ID']
            return image, id, filename
        else:
            label = torch.tensor([row['extent']], dtype=torch.float32)
            #title = f'{row["extent"]}| {row["growth_stage_F"]}, {row["growth_stage_M"]}, {row["growth_stage_S"]}, {row["growth_stage_V"]}'
            #show(image, title=title)
            return image, label, filename

    def augment(self, image):
        if np.random.uniform(0, 1) < 0.5 and 'lr_crop' in self.augment_method: # and image.shape[1]//2 >= self.usize:
            "*** probably needs to be a custom transform to work propperly ***"
            image = self.low_rand_crop(image)
        if 'hflip' in self.augment_method:
            hflip = RandomHorizontalFlip()
            image = hflip(image)
        if (image.shape[1], image.shape[2]) != (self.usize, self.usize):
            resize = Compose([ToPILImage(), Resize((self.usize, self.usize)), ToTensor()])
            image = resize(image)
        if np.random.uniform(0, 1) < 0.5 and 'fourier' in self.augment_method:
            to_pil = ToPILImage()
            image = to_pil(image)
            self.blocker.transform(image)
            image = self.blocker.image
        return image

    def low_rand_crop(self, image):
        c512_resize = Compose([ToPILImage(), Resize((512, 512)), ToTensor()])
        crop = RandomCrop(self.usize)
        image = c512_resize(image)
        image = image[:, image.shape[1]//2:, :]
        return crop(image)


class Fblocker:
    def __init__(self):
        self.image = None
        
    def transform(self, PIL_image):
        image = np.array(PIL_image)
        channels = self.c_split(image)
        shifted_channels = self.shift(channels)
        blocked_channels = self.block_freq(shifted_channels)
        #magnized_channels = self.magnize(blocked_channels)
        ishifted_channels = self.ishift(blocked_channels)
        self.image = self.reassemble(ishifted_channels)
        #plot9(channels, magnized_channels, ishifted_channels)

    def c_split(self, image):
        return [image[:,:,0], image[:,:,1], image[:,:,2]]
    
    def shift(self, channels):
        new_channels = []
        for channel in channels:
            new_channels.append(np.fft.fftshift(np.fft.fft2(channel)))
        return new_channels

    def block_freq(self, channels):
        s = 10
        block = np.zeros((s, s))
        blockorix = channels[0].shape[0]//2 - s//2
        blockoriy = channels[0].shape[1]//2 - s//2

        for channel in channels:
            channel[blockorix:blockorix+s, blockoriy:blockoriy+s] = block
        return channels

    def magnize(self, channels):
        new_channels = []
        for channel in channels:
            new_channels.append(np.log(np.abs(channel) + 1))
        return new_channels
    
    def ishift(self, channels):
        new_channels = []
        for channel in channels:
            new_channels.append(np.abs(np.fft.ifft2(np.fft.ifftshift(channel))))
        return new_channels
    
    def reassemble(self, channels):
        return torch.tensor(np.stack(channels, axis=2), dtype=torch.float32).permute(2, 0, 1)/255