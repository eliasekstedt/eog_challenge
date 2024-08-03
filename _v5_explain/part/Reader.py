
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from part.CustomTransforms import CenterBlock, SegmentBlock

image_size = 224
def get_augmentations(augmentation, mode):
    selected = []
    if True:
        selected += [transforms.Resize(224)]
    if augmentation['rotation'] and mode in ['train', 'sample']:
        selected += [
            transforms.RandomRotation(degrees=(0, 360)),
            #transforms.Resize([image_size[0]]*2)
            ]
    if augmentation['center_block'] > 0 and mode in ['train', 'sample']:
        selected += [CenterBlock(block_chance=augmentation['center_block'], image_size=[4*image_size//5]*2)]
    if augmentation['segment_block'] > 0 and mode in ['train', 'sample']:
        selected += [SegmentBlock(block_chance=augmentation['segment_block'])]
    if augmentation['bright'] and mode in ['train', 'sample']:
        selected += [transforms.ColorJitter(brightness=(0.5, 2.0))]
    if augmentation['hflip'] and mode in ['train', 'sample']:
        selected += [transforms.RandomHorizontalFlip()]
    if augmentation['vflip'] and mode in ['train', 'sample']:
        selected += [transforms.RandomVerticalFlip()]
    if augmentation['center_crop'] and mode in ['train', 'sample', 'eval']:
        selected += [transforms.CenterCrop(4*image_size//5)]
    selected += [transforms.ToTensor()]
    if mode in ['train', 'eval']:
        selected += [transforms.Normalize(mean=[0.4467, 0.4443, 0.3265], std=[0.2368, 0.2382, 0.2752])] # eog_224, class 0 and class 1
    return transforms.Compose(selected)

just_to_tensor = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

class Reader(Dataset):
    def __init__(self, path_csv, path_im, augmentation, mode):
        assert mode in ['train', 'eval', 'sample']
        self.mode = mode
        self.csv = pd.read_csv(path_csv)
        self.path_im = path_im
        self.transforms = get_augmentations(augmentation, mode)

    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        im_name = row['address']
        image = Image.open(f"{self.path_im}{im_name}")
        if self.mode == 'sample': # creating sample.png
            image_aug = self.transforms(image)
            image = just_to_tensor(image)
            return image, image_aug
        elif self.mode == 'eval':
            image = self.transforms(image)
            return im_name, image
        else: # mode == train
            label = torch.tensor([row['label']])
            label = torch.squeeze(label)
            image = self.transforms(image)
            return im_name, image, label