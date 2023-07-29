
import pandas as pd
import torch
from PIL import Image
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Reader(Dataset):
    def __init__(self, path_csv, path_im):
        self.path_im = path_im
        self.map = pd.read_csv(path_csv)#.head(100)

    def __len__(self):
        return len(self.map)
    
    def __getitem__(self, idx):
        row = self.map.iloc[idx]
        name = row['filename']
        #image = Image.open(self.path_im + name)
        #image = self.make_size_uniform(image=image, size=(128, 128))
        image = read_image(self.path_im + name)
        x, y, z = image.shape[0], image.shape[1], image.shape[2]
        return x, y, z

    def make_size_uniform(self, image, size):
        resize = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        image = resize(image)
        image.requires_grad_(True)
        return image


def main():
    if False:
        path = {'labeled':'data/train/',
                'unlabeled':'data/test/',
                'trainmap':'csv/original/Train.csv',
                'valmap':'csv/original/Test.csv'}
        device = 'cuda'
        
        trainset = Reader(path['trainmap'], path['labeled'])
        valset = Reader(path['valmap'], path['unlabeled'])
        trainloader = DataLoader(trainset, batch_size=1, shuffle=False)
        valloader = DataLoader(valset, batch_size=1, shuffle=False)
        x, y, z = [], [], []
        with torch.no_grad():
            for (xx, yy, zz) in trainloader:
                x.append(xx.item())
                y.append(yy.item())
                z.append(zz.item())
        trainsizes = pd.DataFrame({'x':x, 'y':y, 'z':z})
        print('HALF WAY')
        x, y, z = [], [], []
        with torch.no_grad():
            for (xx, yy, zz) in valloader:
                x.append(xx.item())
                y.append(yy.item())
                z.append(zz.item())
        valsizes = pd.DataFrame({'x':x, 'y':y, 'z':z})

        allsizes = pd.concat([trainsizes, valsizes], axis=0)
        print(allsizes['y'].unique())
        print(allsizes['z'].unique())
        print(allsizes['y'].value_counts())
        print(allsizes['x'].value_counts())
        allsizes.to_csv('allsizes.csv', index=False)
    else:
        allsizes = pd.read_csv('allsizes.csv')
        print(allsizes['y'].value_counts())
        print(allsizes['z'].value_counts())

            








if __name__ == '__main__':
    main()