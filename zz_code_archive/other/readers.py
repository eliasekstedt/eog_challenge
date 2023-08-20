


class JustFCReader:
    def __init__(self, path_data):
        self.data = torch.load(path_data).cuda()

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        row = self.data[idx,:]
        features = row[:-2]
        label = row[-2]
        #print(label)
        return features, label

"""
class EvalJustFCReader:
    def __init__(self, path_csv):
        self.map = pd.read_csv(path_csv)

    def __len__(self):
        return len(self.map)
    
    def __getitem__(self, idx):
        row = self.map.iloc[idx]
        context = torch.tensor(self.map.iloc[idx, 3:].tolist(), dtype=torch.float32)
        id = row['ID']
        name = row['filename']
        return context, id, name
"""





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