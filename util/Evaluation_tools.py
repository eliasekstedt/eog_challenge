
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import time
import torch

class Evaluation:
    def __init__(self, hparam, network, testloader, valloader, runpath):
        self.hparam = hparam
        self.network = network
        self.testloader = testloader
        self.valloader = valloader
        self.runpath = runpath
        self.create_testresults_csv('cuda')
        #self.create_submission('cuda')
        self.create_assembly('csv/testsplit.csv')
        self.create_confusion_matrix()
    
    # network.load_state_dict(torch.load(runpath+'model.pth'))
    def create_testresults_csv(self, device):
        print('creating testresults file')
        tic = time.perf_counter()
        preds, labels, fnames = None, None, None
        self.network.eval()
        with torch.no_grad():
            for i, (batch_images, batch_context, batch_labels, batch_fnames) in enumerate(self.testloader):
                #if i%10 == 0:
                #    print(i)
                batch_images = batch_images.to(device)
                batch_context = batch_context.to(device)
                batch_outputs = self.network(batch_images, batch_context)
                batch_outputs = tuple([el[0].item() for el in batch_outputs])
                if preds is None and labels is None and fnames is None:
                    preds, labels, fnames = batch_outputs, batch_labels, batch_fnames
                else:
                    preds = preds + batch_outputs
                    labels = labels + batch_labels
                    fnames = fnames + batch_fnames

        file = pd.DataFrame({'ID':labels, 'extent':preds})
        file['extent'] = file['extent'].clip(lower=0, upper=100)
        print(file)
        file.to_csv(self.runpath+'testresults.csv', index=False)
        toc = time.perf_counter()
        print(f'testresults.csv created in {round(toc-tic, 2)} seconds\n')

    def create_submission(self, device):
        print('creating submission file')
        def get_subfile_name(runpath, hparam):
            subname_components = runpath
            subname_components = [runpath]+[str(param)+'_' for param in hparam.values()]
            subname = ''
            for i in range(len(subname_components)):
                subname = subname + subname_components[i]
            return subname[:-1] + '.csv'
        tic = time.perf_counter()
        self.network.load_state_dict(torch.load(self.runpath+'model.pth'))
        preds, ids, fnames = None, None, None
        self.network.eval()
        with torch.no_grad():
            for i, (batch_images, batch_context, batch_ids, batch_fnames) in enumerate(self.valloader):
                #if i%10 == 0:
                #    print(i)
                batch_images = batch_images.to(device)
                batch_context = batch_context.to(device)
                batch_outputs = self.network(batch_images, batch_context)
                batch_outputs = tuple([el[0].item() for el in batch_outputs])
                if preds is None and ids is None and fnames is None:
                    preds, ids, fnames = batch_outputs, batch_ids, batch_fnames
                else:
                    preds = preds + batch_outputs
                    ids = ids + batch_ids
                    fnames = fnames + batch_fnames

        submission = pd.DataFrame({'ID':ids, 'extent':preds})
        submission['extent'] = submission['extent'].clip(lower=0, upper=100)
        print(submission)
        subfile_name = get_subfile_name(self.runpath, self.hparam)
        submission.to_csv(subfile_name, index=False)
        toc = time.perf_counter()
        print(f'sub-file created in {round(toc-tic, 2)} seconds\n')

    def create_assembly(self, path_extent):
        df_extent = pd.read_csv(path_extent)
        df_pred = pd.read_csv(self.runpath+'testresults.csv')
        df_pred.rename(columns={'extent':'pred'}, inplace=True)
        assembly = pd.concat([df_extent[['ID','filename','extent']], df_pred['pred']], axis=1)
        print(assembly)
        print(assembly['pred'])
        assembly['rpred'] = [round(el/5)*5 for el in assembly['pred']]
        assembly.to_csv(self.runpath+'assembly.csv', index=False)

    def create_confusion_matrix(self):
        assembly = pd.read_csv(self.runpath+'assembly.csv')
        assembly = assembly[assembly['extent'] != 0]
        assembly = assembly[assembly['rpred'] != 0]
        extent_labels = np.sort(assembly["extent"].unique())
        rpred_labels = np.sort(assembly["rpred"].unique())
        print(f'{extent_labels}')
        print(f'{rpred_labels}')
        preds = np.array(assembly['rpred'].to_numpy())
        actuals = np.array(assembly['extent'].to_numpy())
        cmatrix = confusion_matrix(actuals, preds)
        cmatrix_df = pd.DataFrame(cmatrix)#, [i for i in range(5, 105, 5)], [i for i in range(5, 105, 5)])
        plt.figure(figsize=(10,7))
        sn.set(font_scale=1.4)
        sn.heatmap(cmatrix_df, annot=True, annot_kws={'size': 12}, cmap='Blues')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('predicted')
        plt.ylabel('actual')
        plt.savefig(self.runpath+'confusion_matrix.png')
        plt.figure()
        plt.close('all')

