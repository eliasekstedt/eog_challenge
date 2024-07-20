import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def main():
    # Load your dataset
    #data_path = '../../../data/synOC_extended_get_norm_param/rgb/'
    #data_path = '../../../data/ConceptMethod/OC_256/oc/tp_mcd_tuning/'

    data_path = '../../../data/eog_224/'
    """
    import os
    current_path = ''
    pieces = data_path.split('/')
    for piece in pieces[:-1]:
        current_path = f"{current_path}{piece}/"
        print(os.path.exists(current_path), current_path)
    1/0
    """

    dataset = datasets.ImageFolder(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print(len(dataset))
    print(len(dataloader))

    # Calculate mean
    tik = time.perf_counter()
    for images, _ in dataloader:
        for i in range(3):  # Loop through channels
            mean[i] += images[:,i,:,:].mean()
            std[i] += images[:,i,:,:].std()
    tok = time.perf_counter()
    took = tok - tik
    print(f"time: {took//60} minutes")
    mean /= len(dataloader)
    std /= len(dataloader)

    """
    """
    print("Mean: ", mean)
    print("Std: ", std)


if __name__ == '__main__':
    main()