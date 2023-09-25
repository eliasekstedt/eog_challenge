
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


from PIL import Image
from torchvision.transforms import RandomCrop, ToTensor, ToPILImage, Compose, RandomHorizontalFlip, RandomVerticalFlip, Resize

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
    def __init__(self, path_csv, path_im, usize, augment_method=[], crop_ratio=None, crop_freq=None, eval=False):
        self.augment_method = augment_method
        self.crop_ratio = crop_ratio
        self.crop_freq = crop_freq
        self.usize = usize
        self.set = pd.read_csv(path_csv)#.filter(regex='^(?!damage_)', axis=1)
        self.path_im = path_im
        self.eval = eval
        #self.blocker = Fblocker()
        self.hflip = RandomHorizontalFlip()
        self.vflip = RandomVerticalFlip()
        self.ini_resize = Compose([ToPILImage(), Resize((1024, 1024)), ToTensor()])
        self.final_resize = Compose([ToPILImage(), Resize((self.usize, self.usize)), ToTensor()])

    def __len__(self):
        return len(self.set)
    
    def __getitem__(self, idx):
        row = self.set.iloc[idx]
        filename = row['filename']
        context = torch.tensor(self.set.iloc[idx, 3:].tolist(), dtype=torch.float32)
        image = read_image(f'{self.path_im}{filename}')/255
        image = image.type(torch.float32)
        #title = f'{row["damage_DR"]} {row["extent"]}| {row["growth_stage_F"]}, {row["growth_stage_M"]}, {row["growth_stage_S"]}, {row["growth_stage_V"]}'
        #show(image, title=title)
        image = self.augment(image)
        if self.eval:
            id = row['ID']
            return image, context, id, filename
        else:
            label = torch.tensor([row['extent']], dtype=torch.float32)
            #title = f'{row["damage_DR"]} {row["extent"]}| {row["growth_stage_F"]}, {row["growth_stage_M"]}, {row["growth_stage_S"]}, {row["growth_stage_V"]}'
            #show(image, title=title)
            return image, context, label, filename

    def augment(self, image):
        if 'rcrop' in self.augment_method and np.random.uniform(0, 1) < self.crop_freq:
            image = self.rcrop(image)
        if 'hflip' in self.augment_method and np.random.uniform(0, 1) < 0.5:
            image = self.hflip(image)
        if 'vflip' in self.augment_method and np.random.uniform(0, 1) < 0.5:
            image = self.vflip(image)
        if (image.shape[1], image.shape[2]) != (self.usize, self.usize):
            image = self.final_resize(image)
        return image

    def rcrop(self, image):
        crop = RandomCrop((int(image.shape[1]*self.crop_ratio), int(image.shape[2]*self.crop_ratio)))
        #return crop(image[:,int(image.shape[1]*0.25):, :]) # COMPARE (UP TO CR=0.7) 
        return crop(image)







"""

        if np.random.uniform(0, 1) < 0.80 and 'fourier' in self.augment_method:
            to_pil = ToPILImage()
            image = to_pil(image)
            self.blocker.transform(image)
            image = self.blocker.image

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
        height = channels[0].shape[0]
        width = channels[0].shape[1]
        option = '2'
        if option == '0':
            s = 1
            block = np.zeros((s, s))
            blockorix = channels[0].shape[0]//2# - s//2
            blockoriy = channels[0].shape[1]//2# - s//2
            for channel in channels:
                channel[blockorix:blockorix+s, blockoriy:blockoriy+s] = block
            return channels
        elif option == '1':
            s = 50
            zchannels = [np.zeros_like(channels[0]) for _ in range(len(channels))]
            for i, channel in enumerate(channels):
                block = channel[(height-s)//2:(height+s)//2, (width-s)//2:(width+s)//2]
                zchannels[i][(height-s)//2:(height+s)//2, (width-s)//2:(width+s)//2] = block
            return zchannels
        elif option == '2':
            s = 0
            r = np.random.uniform(0, 1)
            if r < 0.25:
                for channel in channels:
                    channel[:height, width//2-1:] = 0
            elif r < 0.50:
                for channel in channels:
                    channel[height//2-s:, :width] = 0
            elif r < 0.75:
                for channel in channels:
                    channel[:height, :width//2+s] = 0
            else:
                for channel in channels:
                    channel[:height//2+s, :width] = 0
            return channels
        elif option == '3':
            r = np.random.uniform(0, 1)
            s = 10
            if r < 0.5:
                for channel in channels:
                    channel[:height, width//2 + s:] = 0
                    channel[:height, :width//2 - s] = 0
            else:
                for channel in channels:
                    channel[height//2 + s:, :width] = 0
                    channel[:height//2 - s, :width] = 0
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
        return torch.tensor(np.stack(channels, axis=2).astype(np.uint8), dtype=torch.float32).permute(2, 0, 1)/255
"""


