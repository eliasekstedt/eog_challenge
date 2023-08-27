
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def gshow(image):
    plt.imshow(image.to("cpu").permute(1, 2, 0), cmap='gray')
    #plt.xticks([])
    #plt.yticks([])
    plt.show()

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
    ax1.plot(epochs, b, testcol, label='count')
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
    #test = pd.read_csv('csv/Val.csv')
    fold_0 = pd.read_csv('csv/fold_0.csv')
    fold_1 = pd.read_csv('csv/fold_1.csv')
    train = pd.concat([fold_0, fold_1], axis=0)

    train['pred'] = [None for _ in range(len(train))]
    reorder = list(train.columns[0:3]) + [train.columns[-1]] + list(train.columns[3:-1])
    train = train[reorder]
    rmse_list = []
    for i in range(101):
        train['pred'] = np.ones(len(train))*i
        rmse = np.sqrt(np.mean((train['extent'] - train['pred'])**2))
        print(f'guess|rmse: {i}|{rmse}')
        rmse_list.append(rmse)
    
    plt.plot(range(len(rmse_list)), rmse_list)
    plt.show()





def test11():
    imsizes = pd.read_csv('zz_code_archive/other/allsizes.csv')
    counts = imsizes.value_counts(['y','z'])
    print(imsizes)
    print(counts)

def test12():
    # heatmap of error contribution
    """
    further improvement:
    * i want to replace the bar plot with a thin red stripe on the relevant row of the heat map to show
    the relative contribution to the total error by datapoints of the corresponding label.
    * i want to add a thin blue stripe to indicate the total number of datapoints belonging to
    corresponding label
    * green to indicate the target value
    * i need to make everything compatible with the actual file that i will read the data from
    """
    def get_heatmap(df):
        error_contribution_heatmap = None
        for extent in df['extent'].unique():
            if error_contribution_heatmap == None:
                error_contribution_heatmap = get_heatmap_row(df, row_number=extent)
            else:
                row = get_heatmap_row(df, row_number=extent)
                error_contribution_heatmap = torch.cat([error_contribution_heatmap, row], dim=1)

        return error_contribution_heatmap/error_contribution_heatmap.max()

    def get_heatmap_row(df, row_number):
        row_width = 100
        row_height = 20
        counts_info = df.loc[df['extent'] == row_number].value_counts('rpred')
        counts_of = [int(i) for  i in counts_info.index]
        counts = [float(i) for i in counts_info.values]
        cm_row = torch.zeros((3, row_height, row_width))
        #print(f'*** {row_number} ***')
        for i, of in enumerate(counts_of):
            count = float(counts[i])
            distance = np.abs(of - row_number)
            #print(f'       count: {count}\n          of: {of}\n        dist: {distance}\ncontribution: {count*distance}\n')
            cm_row[:, :, of] = count*distance
        return cm_row

    def get_distribution(heatmap):
        collection_rows = [heatmap.shape[1]//22 + i*heatmap.shape[1]//11 for i in range(11)]
        distribution = [None]*len(collection_rows)
        for i, coord in enumerate(collection_rows):
            distribution[i] = heatmap[:,coord:coord+1, :].sum().item()
        return [int(round(el/max(distribution)*heatmap.shape[2])) for el in distribution] # length of bar in pixels
    
    def add_vstripe(heatmap):
        row_height = 20
        row_width = 100
        stripe = torch.zeros(3, row_height, 1)
        stripe[1, :, :] = 0.4*heatmap.max()
        print(heatmap.shape)
        crows = [i*row_height for i in range(11)]
        ccols = [i*row_width//10 for i in range(11)]
        print(crows)
        print(ccols)
        #1/0
        for i in range(11):
            rowlow = crows[i]
            rowhigh = rowlow + row_height
            collow = crows[i]//2
            colhigh = collow + 1
            print(rowlow, rowhigh, min(collow,99), min(colhigh,100))
            #print(stripe.shape)
            #print(heatmap[:, rowlow:rowhigh, collow:colhigh].shape)
            heatmap[:, rowlow:rowhigh, min(collow,99):min(colhigh,100)] = stripe



    def add_hstripe(hm, distribution, color):
        print(distribution)
        hm_vulen = hm.shape[1]//11
        row_coords = list(range(hm.shape[1]))[hm_vulen-1::hm_vulen]
        row_coords = [el-1*(color=='blue') for el in row_coords] # positioning red v blue
        for i, coord in enumerate(row_coords):
            stripe_len = distribution[i]
            stripe = torch.zeros(3, 1, stripe_len)
            if color == 'red':
                stripe[0, 0:1, :stripe_len] = 0.4*hm.max()
            else:
                stripe[2, 0:1, :stripe_len] = 0.4*hm.max()
            hm[:, coord:coord+1, :stripe_len] = stripe

    def plot_stats(heatmap):
        fig, ax1 = plt.subplots(1)
        ax1.imshow(heatmap.to("cpu").permute(1, 2, 0), cmap='gray')
        plt.tight_layout()
        plt.show()

    #df = pd.read_csv('run/20_15_38_03_special/eval_data_7.31243.csv')
    df = pd.read_csv('run/20_15_38_03_special/alt_evaldata.csv')
    df.rename(columns={'pred_extent':'pred'}, inplace=True)
    df['rpred'] = df['pred'].round()
    df = df[['extent', 'pred', 'rpred']]
    print(df)

    data = pd.DataFrame({'extent':[10*i for i in range(11)], 'rpred':[99*(np.random.uniform(0, 1)>0.5) for _ in range(11)]})
    for _ in range(50):
        new = pd.DataFrame({'extent':[10*i for i in range(11)], 'rpred':[np.random.uniform(0, 1)*99 for _ in range(11)]})
        data = pd.concat([data, new], axis=0)

    vc = data['extent'].value_counts()
    ivc = list(vc.index)
    vvc = list(vc.values)
    rvvc = [int(round(el/max(vvc)*100)) for el in vvc]

    
    error_contribution_heatmap = get_heatmap(data)
    distribution = get_distribution(error_contribution_heatmap)
    add_vstripe(error_contribution_heatmap)
    #add_hstripe(error_contribution_heatmap, distribution, 'red')
    #add_hstripe(error_contribution_heatmap, rvvc, 'blue')
    plot_stats(error_contribution_heatmap)
    """
    """
    
    

def test13():
    pass


def test14():
    pass



def main():
    test12()
    #test11()
    #test10()
    #test9()
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