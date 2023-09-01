
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import seaborn as sns

def show(image, runpath='', title=''):
    #plt.imshow(image.to("cpu").permute(1, 2, 0))#, cmap='gray')
    plt.imshow(image.to("cpu").detach().permute(1, 2, 0))#, cmap='gray')
    #plt.title(title)
    plt.xticks([])
    plt.yticks([])
    #plt.savefig(runpath+title+'.png')
    #plt.figure()
    #plt.close('all')
    plt.show()



def test0():
    seq = np.random.normal(0, 1, 9)
    seq = tuple(seq.tolist())
    out = [1*(el>0) for el in seq]
    print(out)

def test1():
    left = tuple(np.random.randint(0, 1) for _ in range(2))
    middle = tuple(np.random.randint(0, 1) for _ in range(2))
    right = tuple(np.random.randint(0, 1) for _ in range(3))
    left = left + right
    
def test2():
    evaldata = pd.read_csv('run/initiate_WF1/28_13_05_28/evaldata.csv')
    from sklearn.metrics import confusion_matrix
    cmatrix = confusion_matrix(evaldata['extent'], evaldata['pred'])
    import seaborn as sns
    sns.heatmap(cmatrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'cmatrix')
    plt.figure()
    plt.close('all')

def test3():
    def showme(tp_im, tn_im, fp_im, fn_im):
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        """
        axs[0].imshow(tp_im)
        axs[0].set_title('tp_im')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].imshow(fn_im)
        axs[1].set_title('fn_im')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[2].imshow(fp_im)
        axs[2].set_title('fp_im')
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[3].imshow(tn_im)
        axs[3].set_title('tn_im')
        axs[3].set_xticks([])
        axs[3].set_yticks([])
        plt.tight_layout()
        plt.show()
        """
        axs[0,0].imshow(tp_im)
        axs[0,0].set_title('tp_im')
        axs[0,0].set_xticks([])
        axs[0,0].set_yticks([])
        axs[0,1].imshow(fn_im)
        axs[0,1].set_title('fn_im')
        axs[0,1].set_xticks([])
        axs[0,1].set_yticks([])
        axs[1,0].imshow(fp_im)
        axs[1,0].set_title('fp_im')
        axs[1,0].set_xticks([])
        axs[1,0].set_yticks([])
        axs[1,1].imshow(tn_im)
        axs[1,1].set_title('tn_im')
        axs[1,1].set_xticks([])
        axs[1,1].set_yticks([])
        plt.tight_layout()
        plt.show()


    df = pd.read_csv('evaldata0.csv')
    print(f'tp: {len(df[(df["extent"]==1) & (df["pred"]==1)])}')
    print(f'tn: {len(df[(df["extent"]==0) & (df["pred"]==0)])}')
    print(f'fp: {len(df[(df["extent"]==0) & (df["pred"]==1)])}')
    print(f'fn: {len(df[(df["extent"]==1) & (df["pred"]==0)])}')

    tp_fname = df[(df["extent"]==1) & (df["pred"]==1)]['filename'].to_list()
    tn_fname = df[(df["extent"]==0) & (df["pred"]==0)]['filename'].to_list()
    fp_fname = df[(df["extent"]==0) & (df["pred"]==1)]['filename'].to_list()
    fn_fname = df[(df["extent"]==1) & (df["pred"]==0)]['filename'].to_list()

    #print(fp_fname.tolist())
    impath = '../data/train/'
    for i in range(100):
        #ind = i%min(len(tp_fname), len(tn_fname), len(fp_fname), len(fn_fname))

        tp_im = Image.open(f'{impath}{tp_fname[i%len(tp_fname)]}')
        tp_r, tp_g, tp_b = tp_im.split()
        tn_im = Image.open(f'{impath}{tn_fname[i%len(tn_fname)]}')
        fp_im = Image.open(f'{impath}{fp_fname[i%len(fp_fname)]}')
        fn_im = Image.open(f'{impath}{fn_fname[i%len(fn_fname)]}')
        showme(tp_im, tn_im, fp_im, fn_im)

def test4():
    df = pd.read_csv('evaldata.csv')
    print(df)
    tp = df[(df["extent"]==1) & (df["pred"]==1)][df.columns[5:]]
    tn = df[(df["extent"]==0) & (df["pred"]==0)][df.columns[5:]]
    fp = df[(df["extent"]==0) & (df["pred"]==1)][df.columns[5:]]
    fn = df[(df["extent"]==1) & (df["pred"]==0)][df.columns[5:]]
    print(df.columns)
    print(df.columns[5:])



def test5():
    lst = ['hej', 'hopp']
    print('hej' in lst)
    print('hop' in lst)

def test6():
    
    image_tensor = torch.randn(1, 3, 64, 64)
    target_size = (128, 128)
    resized_image = F.interpolate(image_tensor, size=target_size, mode='bilinear', align_corners=False)

    print("Original size:", image_tensor.shape)
    print("Resized size:", resized_image.shape)
    show(resized_image)

def test7():
    df = pd.read_csv('evaldata.csv')
    tp = df[(df["extent"]==1) & (df["pred"]==1)][df.columns[2:]]
    tn = df[(df["extent"]==0) & (df["pred"]==0)][df.columns[2:]]
    fp = df[(df["extent"]==0) & (df["pred"]==1)][df.columns[2:]]
    fn = df[(df["extent"]==1) & (df["pred"]==0)][df.columns[2:]]
    
    # Combine into one DataFrame
    tps = tp.sum()
    tns = tn.sum()
    fps = fp.sum()
    fns = fn.sum()
    summed_dfs = [tps, tns, fps, fns]
    all_data = pd.concat(summed_dfs, keys=['Group1', 'Group2', 'Group3', 'Group4']).reset_index(level=0).rename(columns={"level_0": "Group"})

    # Melt into long-form
    long_data = pd.melt(all_data, id_vars=["Group"], value_vars=tp.sum().index.to_list(), var_name="Column", value_name="Sum")

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Column", y="Sum", hue="Group", data=long_data)
    plt.title('Grouped Bar Plot')
    plt.show()
    """
    colnames = tp.sum().index.to_list()
    tps = tp.sum().values.tolist()
    tns = tn.sum().values.tolist()
    fps = fp.sum().values.tolist()
    fpn = fn.sum().values.tolist()
    """


def test8():

    df = pd.read_csv('evaldata.csv')
    tp = df[(df["extent"]==1) & (df["pred"]==1)][df.columns[5:]]
    tn = df[(df["extent"]==0) & (df["pred"]==0)][df.columns[5:]]
    fp = df[(df["extent"]==0) & (df["pred"]==1)][df.columns[5:]]
    fn = df[(df["extent"]==1) & (df["pred"]==0)][df.columns[5:]]
    
    # Combine into one DataFrame
    tps = pd.DataFrame(tp.mean()) 
    tns = pd.DataFrame(tn.mean())
    fps = pd.DataFrame(fp.mean())
    fns = pd.DataFrame(fn.mean())
    sdfs = [tps, tns, fps, fns]
    print(sdfs)

    # Example summed DataFrames
    summed_dfs = [pd.DataFrame({"col1":[5], "col2":[10]}), pd.DataFrame({"col1":[4], "col2":[12]}),
                pd.DataFrame({"col1":[6], "col2":[11]}), pd.DataFrame({"col1":[7], "col2":[9]})]
    
    # Combine into one DataFrame
    all_data = pd.concat(summed_dfs, keys=['Group1', 'Group2', 'Group3', 'Group4']).reset_index(level=0).rename(columns={"level_0": "Group"})
    all_data2 = pd.concat(sdfs, keys=['Group1', 'Group2', 'Group3', 'Group4']).reset_index(level=0).rename(columns={"level_0": "Group"})
    print(all_data)
    print(all_data2)
    1/0


    # Melt into long-form
    long_data = pd.melt(all_data, id_vars=["Group"], value_vars=["col1", "col2"], var_name="Column", value_name="Sum")
    long_data = pd.melt(all_data2, id_vars=["Group"], value_vars=["col1", "col2"], var_name="Column", value_name="Sum")

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Column", y="Sum", hue="Group", data=long_data)
    plt.title('Grouped Bar Plot')
    plt.show()

def test9():
    # create a sample dataset
    data = {'A': [1, 2, 3, 4, 5], 
            'B': [10, 20, 30, 40, 50]}
    df = pd.DataFrame(data)
    # group the rows by column A
    grouped = df.groupby('A')
    # calculate the mean of each column for each group
    means = grouped.mean()
    # create a new dataframe with the means
    result = pd.DataFrame(means, columns=df.columns)
    print(result)

    df = pd.read_csv('evaldata.csv')

    pos = df[df['extent']==1]
    gpos = pos.groupby((pos['pred']==1))
    gpmeans = gpos.mean()
    gpdf = pd.DataFrame(gpmeans, columns=df.columns)
    gpdf['category'] = ['fn', 'tp']

    neg = df[df['extent']==0]
    gneg = neg.groupby((neg['pred']==1))
    gnmeans = gneg.mean()
    gndf = pd.DataFrame(gnmeans, columns=df.columns)
    gndf['category'] = ['tn', 'fp']
    df = pd.concat([gpdf, gndf], axis=0)
    df = df[df.columns[5:]].reset_index()
    df = df[['category'] + df.columns[1:-1].tolist()]
    print(df)
    1/0

    gdf = df.groupby((df['extent']==1) & (df['pred']==1))
    means = gdf.mean()
    result = pd.DataFrame(means, columns=df.columns)
    print(result)

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

def test15():
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





"""
####
            fname = 'L430F01265C39S13909Rp40378.jpg'
            if fname in _:
                ind = _.index(fname)
                print(ind)
                show(batch_images[ind])
                #1/0
            ####
"""