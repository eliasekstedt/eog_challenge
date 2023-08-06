
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def test0():
    def get_subfile_name(runpath, hparam):
        subname_components = runpath
        subname_components = [runpath]+[str(param)+'_' for param in hparam.values()]
        subname = ''
        for i in range(len(subname_components)):
            subname = subname + subname_components[i]
        return subname[:-1] + '.csv'


    hparam = {'batch_size': 200,
            'nr_epochs': 15,
            'architecture_name':'res18fc',
            'weight_decay': 1e-7,
            'dropout_rate': 0.0,
            'resizes':(128, 128)}

    runpath = '' #'run/00_00_00_00/'
    subname = get_subfile_name(runpath, hparam)

    df = pd.read_csv('allsizes.csv')

    df.to_csv(subname, index=False)

def test1():
    a = np.array([4, 5, 6, 6, 8])
    b = np.array([6, 7, 4, 7, 9])

    epochs = range(1, len(a) + 1)
    traincol = 'tab:blue'
    testcol = 'tab:red'
    # figure
    fig, ax1 = plt.subplots(1,figsize=(6, 8))
    # cost
    ax1.plot(epochs, a, traincol, label='train')
    ax1.plot(epochs, b, testcol, label='val')
    ax1.set_ylim([4, 15])
    ax1.legend()
    #ax1.set_xticks([])
    ax1.set_ylabel('Cost')
    plt.tight_layout()
    plt.show()

def test2():
    c1 = torch.load('c1_tensor.pt')
    print(c1)
    print(torch.mean(c1))

def test3():
    trainsplit = pd.read_csv('csv/trainsplit.csv')
    #testsplit = pd.read_csv('csv/save_testsplit.csv')
    context1 = torch.tensor(trainsplit.iloc[0, 3:].tolist(), dtype=torch.float32)
    #context2 = torch.tensor(testsplit.iloc[0, 2:].tolist(), dtype=torch.float32)
    print(context1)
    #print(context2)
    #trainsplit = trainsplit[trainsplit.columns[~trainsplit.columns.str.startswith('damage')]]
    #testsplit = testsplit[testsplit.columns[~testsplit.columns.str.startswith('damage')]]
    #trainsplit.to_csv('csv/trainsplit.csv', index=False)
    #testsplit.to_csv('csv/testsplit.csv', index=False)


def test4():
    device = 'cuda'
    path = {'trainmap':'csv/save_trainsplit.csv',
            'testmap':'csv/save_testsplit.csv',
            'valmap':'csv/Val.csv',
            'data_labeled':'data/train/',
            'data_unlabeled':'data/test/'
            }
    hparam = {'batch_size': 256,
            'nr_epochs': 10,
            'architecture_name':'res18fc',
            'weight_decay': 1e-7,
            'dropout_rate': 0.0,
            'resizes':(128, 128),
            'penalty':0}
    

    runpath = 'run/testing/06_08_18_53/'

    from util.Networks import Res18FCNet
    from util.Readers import Res18FCReader as Reader
    testset = Reader(path['testmap'], path['data_labeled'], resizes=hparam['resizes'])
    testloader = DataLoader(testset, batch_size=hparam['batch_size'], shuffle=False)
    network = Res18FCNet(hparam['architecture_name'], hparam['weight_decay'], hparam['dropout_rate']).to(device)
    from util.Evaluation_tools import create_pseudo_submission
    create_pseudo_submission(network, runpath, testloader, device)

def test5():
    pass

def test6():
    pass

def test7():
    pass

def main():
    test4()
    #test3()
    #test2()
    #test1()
    #test0()









if __name__ == '__main__':
    main()