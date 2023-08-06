
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

import numpy as np

class EvalpreAnalysisReader(Dataset):
    def __init__(self, path_csv, path_im, resizes):
        self.resizes = resizes
        self.path_im = path_im
        self.map = pd.read_csv(path_csv)

    def __len__(self):
        return len(self.map)
    
    def __getitem__(self, idx):
        row = self.map.iloc[idx]
        context = torch.tensor(self.map.iloc[idx, 3:].tolist(), dtype=torch.float32)
        id = row['ID']
        name = row['filename']
        image = Image.open(self.path_im+name)
        image = self.make_size_uniform(image=image, size=self.resizes)
        # skycrop
        #image = image[:,image.shape[1]//2:, :]
        return image, context, id, name

    def make_size_uniform(self, image, size):
        resize = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        image = resize(image)
        image.requires_grad_(True)
        return image

def distplot(df):
    distributions = [None]*11
    for i in range(len(distributions)):
        distributions[i] = list(df.loc[df['extent']==i*10]['pred'])
    distributions = distributions[1:]
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, figsize=(8, 16))
    for i, ax in enumerate((ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10)):
        ax.hist(distributions[i])
        ax.set_xlim([0, 100])
        ax.set_ylabel(f'{10*(i+1)},{len(distributions[i])}')
        if i < 9:
            ax.set_xticks([])
    plt.savefig(f'preAnalysis/distplot.png')
    plt.figure()
    plt.close('all')
    #plt.show()

def scatter_matrix(df):
    sns.set(style="ticks")
    sns.pairplot(df)
    plt.savefig(f'preAnalysis/scatter_matrix.png')
    plt.figure()
    plt.close('all')
    #plt.show()

def create_confusion_matrix(class_results, runpath):
    import numpy as np
    actuals = np.array(class_results['extent'].to_numpy())
    preds = np.array(class_results['rpred'].to_numpy())
    from sklearn.metrics import confusion_matrix
    cmatrix = confusion_matrix(actuals, preds)
    cmatrix_df = pd.DataFrame(cmatrix, [i for i in range(5, 105, 5)], [i for i in range(5, 105, 5)])
    import seaborn as sn
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)
    sn.heatmap(cmatrix_df, annot=True, annot_kws={'size': 12}, cmap='Blues')
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.savefig(runpath+'confusion_matrix.png')
    plt.figure()
    plt.close('all')

def main():
    runpath = 'preAnalysis/'
    device = 'cuda:0'
    hparam = {'batch_size': 128,
            'nr_epochs': 20,
            'architecture_name':'res18fc',
            'weight_decay': 1e-7,
            'dropout_rate': 0.0,
            'resizes':(256, 256)}

    path = {'trainmap':'csv/trainsplit.csv',
            'testmap':'csv/testsplit.csv',
            'valmap':'csv/Val.csv',
            'data_labeled':'data/train/',
            'data_unlabeled':'data/test/',
            'pred_data': 'preAnalysis/pred_data.csv'
            }
    
    


    # take the hotncode test set and generate predictions with the most recent model
    done = True
    if not done:
        from util.Networks import Res18FCNet as Net
        network = Net(hparam['architecture_name'], hparam['weight_decay'], hparam['dropout_rate']).to(device)
        valset = EvalpreAnalysisReader(path['testmap'], path['data_labeled'], resizes=hparam['resizes'])
        #valset = EvalpreAnalysisReader(path['valmap'], path['data_unlabeled'], resizes=hparam['resizes'])
        valloader = DataLoader(valset, batch_size=hparam['batch_size'], shuffle=False)
        from util.Tools import create_submission
        create_submission(network, runpath, valloader, hparam, device)

    # extract the predictions and labels as lists
    pred_data = pd.read_csv(path['pred_data'])
    pred_data.rename(columns={'extent':'pred'}, inplace=True)
    test_data = pd.read_csv(path['testmap'])
    assembled = pd.concat([test_data[['ID','filename','extent']], pred_data['pred']], axis=1)
    assembled['rmse'] = torch.sqrt(torch.tensor((assembled['pred'] - assembled['extent'])**2))
    #assembled['rpred'] = [round(el/10)*10 for el in assembled['pred']]
    assembled['rpred'] = [round(el/5)*5 for el in assembled['pred']]
    assembled['diff'] = assembled['extent'] - assembled['pred']
    #assembled.sort_values('rmse', inplace=True, ascending=False)
    print(assembled['rmse'].sum()/np.sqrt(len(assembled)))
    1/0
    print(assembled.loc[assembled['extent']>0])
    df = assembled.loc[assembled['extent'] == 10]['rpred']
    print(df)
    
    print(df.value_counts('preds'))
    print(len(df))


    plot_smatrix, plot_dist, plot_cmatrix = False, False, True
    # plot the distributions
    if plot_smatrix:
        scatter_matrix(assembled.loc[assembled['extent'] > 0][['extent', 'pred', 'rmse']])

    if plot_dist:
        distplot(assembled)

    if plot_cmatrix:
        create_confusion_matrix(assembled[(assembled['extent'] > 0) & (assembled['rpred'] > 0)], 'preAnalysis/')
        #create_confusion_matrix(assembled, 'preAnalysis/')


        
if __name__ == '__main__':
    main()






















#data = data.drop(data.columns[data.columns.str.startswith('damage')], axis=1)

