
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
    from util.Evaluation_tools import create_assembly
    create_pseudo_submission(network, runpath, testloader, device)
    create_assembly()


def test5():
    print(list(range(100))[::10])

def test6():
    # better save decisions
    df = pd.read_csv('bsd_data.txt')
    df = df[['COST', 'cost']]
    set0 = df[:12]
    set1 = df[12:24]
    set2 = df[24:36]
    set3 = df[36:48]

    sets = [set0, set1, set2, set3]
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, figsize=(8, 16))
    for i, ax in enumerate((ax0, ax1, ax2, ax3)):
        ax.plot(range(len(sets[i])), sets[i])
        if i < 3:
            ax.set_xticks([])
        #ax.plot(range(len(set1)), set1)
        #ax.plot(range(len(set2)), set2)
        #ax.plot(range(len(set3)), set3)
    plt.tight_layout()
    plt.show()

def test7():
    def get_standard_savepoints(set):
        x = [epoch for epoch in range(len(set0)) if set0['cost'][epoch] == min(set0['cost'][:epoch+1])]
        y = [set0['cost'][epoch] for epoch in range(len(set0)) if set0['cost'][epoch] == min(set0['cost'][:epoch+1])]
        return x, y
    # better save decisions
    df = pd.read_csv('bsd_data.txt')
    df = df[['COST', 'cost']]
    set0 = df[:12]
    set1 = df[12:24]
    set2 = df[24:36]
    set3 = df[36:48]

    savepoints_x, savepoints_y = get_standard_savepoints(set3)
    #savepoints = [(epoch, set0['cost'][epoch]) for epoch in range(len(set0)) if set0['cost'][epoch]==min(list(set0['cost'])[:epoch])]
    plt.plot(range(len(set0)), set0)
    plt.scatter(savepoints_x, savepoints_y)
    plt.show()

def test8():
    f0 = pd.read_csv('csv/fold_0.csv')
    f0 = f0[f0.columns[3:]]
    f0_cats = list(f0.columns)
    f0_colsums = np.array(f0.sum())
    f1 = pd.read_csv('csv/fold_1.csv')
    f1 = f1[f1.columns[3:]]
    f1_cats = list(f1.columns)
    f1_colsums = np.array(f1.sum())
    
    colsumdiff = f1_colsums - f0_colsums
    print(colsumdiff)
    plt.bar(range(len(colsumdiff)), colsumdiff)
    plt.show()

def test9():
    lim = 0.9
    method = 2
    nr_epochs = 20
    for i in range(1, nr_epochs + 1):
        rn = np.random.normal(0, .5)
        print(i, rn)
        if method == 0 and (rn < -lim or rn > lim):
            print('method 1: break')
            break
        if method == 1 and (rn < -lim or rn > lim):
            print('method 2: continue')
            continue
        if method == 2 and (rn < -lim or rn > lim):
            print('method 3: nr_epochs + 1')
            i = nr_epochs + 1
    
    print('\nif this plays the mashine continues with what comes next')




def test10():
    pass

def test11():
    pass

def test12():
    pass

def test13():
    pass

def test14():
    pass



def main():
    test9()
    #test8()
    #test7()
    #test6()
    #test5()
    #test4()
    #test3()
    #test2()
    #test1()
    #test0()









if __name__ == '__main__':
    main()


















#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycircos
Garc = pycircos.Garc
Gcircle = pycircos.Gcircle

def get_circle_garc(df):
    circle = Gcircle(figsize=(8,8))
    circle = Gcircle(figsize=(8,8))
    for i in range(len(df)):
        name = df.iloc[i, 0]
        name = df.iloc[i, 0]
        length = df.iloc[i, -1]
        length = df.iloc[i, -1]
        arc = Garc(arc_id=name, size=length, interspace=2, raxis_range=(935, 935), labelposition=80, label_visible=True)
        arc = Garc(arc_id=name, size=length, interspace=2, raxis_range=(935, 985), labelposition=80, label_visible=True)
        circle.add_garc(arc)
        circle.add_garc(arc) 

    circle.set_garcs(-65,245)
    circle.set_garcs(-65,245) 

    for arc_id in circle.garc_dict:
        circle.tickplot(arc_id, raxis_range=(985,1000), tickinterval=20000000, ticklabels=None)
        circle.tickplot(arc_id, raxis_range=(985,1000), tickinterval=20000000, ticklabels=None)
    return circle


def get_circle(df):
    for i in range(len(df)):


    for arc_id in circle.garc_dict:
    return circle
"""