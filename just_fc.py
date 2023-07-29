
import pandas as pd
import torch

from torch.utils.data import DataLoader


class Reader:
    def __init__(self, path_csv):
        self.map = pd.read_csv(path_csv)






def main():
    path = {'trainmap':'csv/trainsplit.csv',
            'testmap':'csv/testsplit.csv'
            }
    
    hparam = {'batch_size': 200,}
    
    trainset = Reader(path['trainmap'])
    testset = Reader(path['testmap'])
    trainloader = DataLoader(trainset, )






if __name__ == '__main__':
    main()