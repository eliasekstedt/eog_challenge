
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

#from sklearn.metrics import confusion_matrix
#import seaborn as sn
#import time

class Evaluator:
    def __init__(self, runpath, model, reader, path, hparam, device):
        self.runpath = runpath
        set = reader(path['fold_1'], path['data_labeled'], resizes=hparam['resizes'], augment=False, eval=True)
        loader = DataLoader(set, batch_size=hparam['batch_size'], shuffle=False)
        preddata = self.evaluate(model, loader, device)
        self.evaldata = self.assemble_evaldata(path, preddata)
        self.rmse = np.round(np.sqrt(np.mean((self.evaldata['extent'] - self.evaldata['pred_extent'])**2)), 5)

    def evaluate(self, network, loader, device):
        with torch.no_grad():
            preds, ids, fnames = None, None, None
            for i, (batch_images, batch_context, batch_ids, batch_fnames) in enumerate(loader):
                if i%10 == 0:
                    print(i)
                batch_images = batch_images.to(device)
                batch_context = batch_context.to(device)
                batch_outputs = network(batch_images, batch_context)
                batch_outputs = tuple([el[0].item() for el in batch_outputs])
                if preds is None and ids is None and fnames is None:
                    preds, ids, fnames = batch_outputs, batch_ids, batch_fnames
                else:
                    preds = preds + batch_outputs
                    ids = ids + batch_ids
                    fnames = fnames + batch_fnames
            df = pd.DataFrame({'ID':ids, 'extent':preds})
            df['extent'] = df['extent'].clip(lower=0, upper=100)
            return df

    def assemble_evaldata(self, path, preddata):
        preddata.rename(columns={'extent':'pred_extent'}, inplace=True)
        folddata = pd.read_csv(path['fold_1'])
        evaldata = preddata.merge(folddata, on='ID', how='inner')
        evaldata['error'] = np.abs(evaldata['extent'] - evaldata['pred_extent'])
        print(evaldata.columns)
        print(type(evaldata.columns))
        evaldata = evaldata[['ID', 'filename', 'extent', 'pred_extent', 'error']+list(evaldata.columns[4:-1])]
        return evaldata


"""
class CrossEvaluator:
    def __init__(self, runpath, models, reader, path, hparam, device):        
        self.runpath = runpath
        # we dont want to make augmentations and so cant reuse loaders used to train (and also eval=True)
        set_0 = reader(path['fold_0'], path['data_labeled'], resizes=hparam['resizes'], augment=False, eval=True)
        loader_0 = DataLoader(set_0, batch_size=hparam['batch_size'], shuffle=False)
        set_1 = reader(path['fold_1'], path['data_labeled'], resizes=hparam['resizes'], augment=False, eval=True)
        loader_1 = DataLoader(set_1, batch_size=hparam['batch_size'], shuffle=False)
        # do crosswise evaluation
        preddata_0 = self.evaluate(models[1], loader_0, device)
        preddata_1 = self.evaluate(models[0], loader_1, device)
        # assembling dataframes
        self.evaldata = self.assemble_evaldata(path, preddata_0, preddata_1)
        self.rmse = np.round(np.sqrt(np.mean((self.evaldata['extent'] - self.evaldata['pred_extent'])**2)), 5)
        
    def evaluate(self, network, loader, device):
        with torch.no_grad():
            preds, ids, fnames = None, None, None
            for i, (batch_images, batch_context, batch_ids, batch_fnames) in enumerate(loader):
                if i%10 == 0:
                    print(i)
                batch_images = batch_images.to(device)
                batch_context = batch_context.to(device)
                batch_outputs = network(batch_images, batch_context)
                batch_outputs = tuple([el[0].item() for el in batch_outputs])
                if preds is None and ids is None and fnames is None:
                    preds, ids, fnames = batch_outputs, batch_ids, batch_fnames
                else:
                    preds = preds + batch_outputs
                    ids = ids + batch_ids
                    fnames = fnames + batch_fnames
            df = pd.DataFrame({'ID':ids, 'extent':preds})
            df['extent'] = df['extent'].clip(lower=0, upper=100)
            return df

    def assemble_evaldata(self, path, preddata_0, preddata_1):
        assembled_preddata = pd.concat([preddata_0, preddata_1], axis=0)
        assembled_preddata.rename(columns={'extent':'pred_extent'}, inplace=True)
        folddata_0 = pd.read_csv(path['fold_0'])
        folddata_1 = pd.read_csv(path['fold_1'])
        assembled_folddata = pd.concat([folddata_0, folddata_1], axis=0)
        evaldata = assembled_preddata.merge(assembled_folddata, on='ID', how='inner')
        evaldata['error'] = np.abs(evaldata['extent'] - evaldata['pred_extent'])
        print(evaldata.columns)
        print(type(evaldata.columns))
        evaldata = evaldata[['ID', 'filename', 'extent', 'pred_extent', 'error']+list(evaldata.columns[4:-1])]
        return evaldata


"""
















"""
class Evaluation:
    def __init__(self, hparam, network, testloader, valloader, runpath):
        self.hparam = hparam
        self.network = network
        self.testloader = testloader
        self.valloader = valloader
        self.runpath = runpath
        self.create_submission('cuda')
        self.create_testresults_csv('cuda')
        self.create_assembly('csv/testsplit.csv')
        self.create_confusion_matrix()
    
    # network.load_state_dict(torch.load(runpath+'model.pth'))
    def create_testresults_csv(self, device):
        print('creating testresults file')
        tic = time.perf_counter()
        self.network.load_state_dict(torch.load(self.runpath+'model21.pth'))
        preds, labels, fnames = None, None, None
        self.network.eval()
        with torch.no_grad():
            for batch_images, batch_context, batch_labels, batch_fnames in self.testloader:
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
        self.network.load_state_dict(torch.load(self.runpath+'model21.pth'))
        preds, ids, fnames = None, None, None
        self.network.eval()
        with torch.no_grad():
            for batch_images, batch_context, batch_ids, batch_fnames in self.valloader:
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
        #assembly = assembly[assembly['extent'] != 0]
        #assembly = assembly[assembly['rpred'] != 0]
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
"""

