
from datetime import datetime
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
    df = pd.read_csv('evaldata.csv')
    title = f'fn: {len(df[(df["extent"]==1) & (df["pred"]==0)])}\ntp: {len(df[(df["extent"]==1) & (df["pred"]==1)])}\ntn: {len(df[(df["extent"]==0) & (df["pred"]==0)])}\nfp: {len(df[(df["extent"]==0) & (df["pred"]==1)])}'
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

    categories = ['gsf', 'gsm', 'gss', 'gsv', 'ddr', 'dds', 'dg', 'dnd', 'dps', 'dwd', 'dwn', 'sl0', 'sl1', 'sr0', 'sr1']
    vals = np.array(df[df.columns[1:]]).T

    barwidth = 0.15
    positions = [np.arange(len(categories))]
    for i in range(3):
        positions.append(positions[-1] + barwidth)

    plt.figure(figsize=(16, 8))
    for i, pos in enumerate(positions):
        plt.bar(pos, vals[:,i], width=barwidth, label=df.category[i])
    plt.title(title)
    plt.xticks(positions[0] + 1.5*barwidth, categories)
    plt.legend()
    plt.show()

def test10():
    print(str(datetime.now())[8:10])

def test11():
    # figuring out what goes wrong with the barplot
    #d = pd.read_csv('CSV/Val.csv')
    #print(d.columns)
    runpath = 'run/WF1_08/08_11_25_53/'
    from WF1_classifier.parts.Tools import plot_by_ctx_feature
    plot_by_ctx_feature(runpath)


def test12():
    def least_squares(x, X, y):
        theta = np.linalg.pinv(X.T@X)@X.T@y
        return theta

    def blr_approach(X, y, sigma):
        def get_is0(s0):
            return np.linalg.pinv(s0)
        def get_betaXTX(beta, X):
            return beta*X.T@X
        def get_sN(is0, betaXTX):
            return np.linalg.pinv(is0 + betaXTX)
        def get_is0m0(is0, m0):
            return is0@m0
        def get_betaXTy(beta, X, y):
            return beta*X.T@y
        def get_mN(sN, is0m0, betaXTy):
            return sN@(is0m0 + betaXTy)
        
        beta = 1/sigma**2
        s0 = np.array([[1, 0], [0, 1]])
        m0 = np.array([[0], [0]])

        is0 = get_is0(s0)
        betaXTX = get_betaXTX(beta, X)
        sN = get_sN(is0, betaXTX)
        is0m0 = get_is0m0(is0, m0)
        betaXTy = get_betaXTy(beta, X, y)
        mN = get_mN(sN, is0m0, betaXTy)
        return sN, mN

    df = pd.read_csv('gp_test.txt', sep=', ', header=None)#.head(9)
    df.columns = ['accuracy', 'wd', 'time']
    x = np.array(df['wd'].to_list())
    ###
    #x = np.array(range(5))
    #y = [xx*np.random.uniform(-1, 1) for xx in x]
    #y = np.array([[yy] for yy in y])
    ###
    X = np.array([[1, xx] for xx in x])
    y = np.array([[yy] for yy in df['accuracy']])
    #theta = least_squares(x, X, y)

    sigma = 1 ## test
    sN, mN = blr_approach(X, y, sigma)
    
    #xH = np.linspace(x.min(), x.max(), 100)
    print(x.max())
    xH = np.linspace(0, 1, 100)
    XH = np.array([[1, xx] for xx in xH])
    #yH = (XH@theta).flatten()

    mH = (XH@mN).flatten()
    sH = np.diag(XH@sN@XH.T) + sigma**2
    print(mN)
    print(sN)
    #1/0

    xlim = [1e-6, 1e-4]
    ylim = [0.8, 1.2]
    plt.plot(x, y, 'o', color='black')
    #plt.plot(xH, yH)
    #plt.xlim(xlim)
    #plt.ylim(ylim)

    plt.plot(xH, mH, color='black')
    plt.fill_between(xH, mH-1*sH, mH+1*sH, alpha=0.3, color='blue')
    plt.fill_between(xH, mH-2*sH, mH+2*sH, alpha=0.3, color='blue')
    plt.show()


def test13():
    logpath = 'gp_logs/cr_09_09_12_24/gp_log.txt'
    df = pd.read_csv(logpath, sep='\t')
    print(df)
    l_bound, u_bound = 0, 1
    from WF1_classifier.parts.Tools import gp_plot
    gp_plot(df, [l_bound, u_bound], logpath)

def test14():
    pass

def test15():
    pass


def main():
    test13()
    #test12()
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