
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

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
        fig, axs = plt.subplots(4, figsize=(10, 10))
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
        """
        """
        """

    df = pd.read_csv('evaldata.csv')
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
        #tn_im = Image.open(f'{impath}{tn_fname[i%len(tn_fname)]}')
        #fp_im = Image.open(f'{impath}{fp_fname[i%len(fp_fname)]}')
        #fn_im = Image.open(f'{impath}{fn_fname[i%len(fn_fname)]}')
        #showme(tp_im, tn_im, fp_im, fn_im)

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
    pass

def test8():
    pass


def main():
    #test8()
    #test7()
    test6()
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