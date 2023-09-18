
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch



class Evaluator:
    def __init__(self, runpath, model, loader, foldpath, device):
        preddata = self.evaluate(model, loader, device)
        evaldata = self.assemble_evaldata(foldpath, preddata)
        evaldata.to_csv(f'{runpath}evaldata.csv', index=False)
        self.cmatrix = self.get_cmatrix(evaldata)
        self.plot_cmatrix(runpath)

    def evaluate(self, model, loader, device):
        model.eval()
        with torch.no_grad():
            preds, ids = None, None
            for i, (batch_image, batch_ids, _) in enumerate(loader):
                batch_image = batch_image.to(device)
                batch_outputs = model(batch_image)
                batch_outputs = tuple([el[0].item() for el in batch_outputs])
                batch_preds = tuple(1*(logit>0) for logit in batch_outputs)
                if preds is None and ids is None:
                    preds, ids = batch_preds, batch_ids
                else:
                    preds = preds + batch_preds
                    ids = ids + batch_ids
            df = pd.DataFrame({'ID':ids, 'pred':preds})
            return df

    def assemble_evaldata(self, foldpath, preddata):
        folddata = pd.read_csv(foldpath)
        evaldata = preddata.merge(folddata, on='ID', how='inner')
        evaldata['error'] = np.abs(evaldata['extent'] - evaldata['pred'])
        evaldata = evaldata[['ID', 'filename', 'extent', 'pred', 'error']+list(evaldata.columns[4:-1])]
        return evaldata
    
    def get_cmatrix(self, evaldata):
        return confusion_matrix(evaldata['extent'], evaldata['pred'])
    
    def plot_cmatrix(self, runpath):
        sns.heatmap(self.cmatrix, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'{runpath}cmatrix')
        plt.figure()
        plt.close('all')
    



class Heatmap:
    def __init__(self, runpath, height=220, vlines=True):
        self.height, self.width = height, 101
        self.data = self.get_data(runpath)
        self.heatmap = self.get_heatmap()
        if vlines:
            self.get_vline_map()
    
    def get_vline_map(self):
        vline = torch.zeros(3, self.height//11, 1)
        vline[1, :,:] = 0.4*self.heatmap.max()
        extent_levels = self.get_extent_levels()
        v_low = [self.height//11*low//10 for low in extent_levels]
        v_high = [low + self.height//11 for low in v_low]
        h_low = [self.width//10*low//10 for low in extent_levels]
        h_high = [low + 1 for low in h_low]
        for i in range(len(extent_levels)):
            self.heatmap[:, v_low[i]:v_high[i], h_low[i]:h_high[i]] = vline
        
    def get_heatmap(self):
        extent_levels = self.get_extent_levels()
        heatmap = None
        for extent_level in extent_levels:
            if heatmap == None:
                heatmap = self.get_heatmap_row(extent_level)
            else:
                hm_row = self.get_heatmap_row(extent_level)
                heatmap = torch.cat([heatmap, hm_row], dim=1)
        return heatmap/heatmap.max()

    def get_heatmap_row(self, extent_level):
        def get_pred_dist_by_extent(extent_level):
            on_extent_level = self.data.loc[self.data['extent'] == extent_level].value_counts('pred')
            pred_levels = [int(i) for i in on_extent_level.index]
            counts_by_pred_level = [float(i) for i in on_extent_level.values]
            return pred_levels, counts_by_pred_level
        
        pred_levels, counts_by_pred_level = get_pred_dist_by_extent(extent_level)
        row_height = self.height//11
        hm_row = torch.zeros(3, row_height, self.width)
        for i, pred_level in enumerate(pred_levels):
            intensity = np.abs(pred_level - extent_level)*counts_by_pred_level[i]
            hm_row[:,:, pred_level] = intensity
        return hm_row

    def get_data(self, runpath):
        data = pd.read_csv(runpath+'evaldata.csv')
        data = data[['extent', 'pred']]
        data['pred'] = data['pred'].round()
        return data

    def get_extent_levels(self):
        extent_levels = self.data['extent'].unique()
        extent_levels.sort()
        return extent_levels
    
    def save(self, runpath):
        plt.imshow(self.heatmap.to("cpu").permute(1, 2, 0), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'{runpath}heatmap.png')
        plt.figure()
        plt.close('all')
        plt.show()


"""
runpath = 'run/test/26_08_17_47/'
hm = Heatmap(runpath, hlines=True, vlines=True)
ul = hm.heatmap[0, :5, :5]
ur = hm.heatmap[0, :5, 96:]
ll = hm.heatmap[0, 215:, :5]
lr = hm.heatmap[0, 215:, 96:]
corners = torch.cat([torch.cat([ul, ur], dim=1), torch.cat([ll, lr], dim=1)], dim = 0).unsqueeze(0)
corners.to(torch.int64)
print(corners.shape)
corners = torch.cat([corners]*3, dim=0)
#print(corners[0, :,:])
gshow(hm.heatmap)
#gshow(corners)
#for corner in [ul, ur, ll, lr]:
#    print(corner.shape)

#print(hm.heatmap[0, 19:21, 95:])
#print(hm.heatmap[0, 19:21, :5])
#print(hm.heatmap.shape)
data = pd.DataFrame({'extent':[10*i for i in range(11)], 'pred':[100*(np.random.uniform(0, 1)>0.5) for _ in range(11)]})
        for _ in range(5000):
            new = pd.DataFrame({'extent':[10*i for i in range(11)], 'pred':[np.random.uniform(0, 1)*100 for _ in range(11)]})
            data = pd.concat([data, new], axis=0)
"""