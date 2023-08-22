
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

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


























