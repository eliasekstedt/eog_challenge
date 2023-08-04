
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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
    pass

def test3():
    pass

def test4():
    pass

def test5():
    pass

def test6():
    pass

def test7():
    pass

def main():
    test1()
    #test0()









if __name__ == '__main__':
    main()