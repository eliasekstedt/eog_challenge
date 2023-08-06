
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import time
import torch

def create_pseudo_submission(network, runpath, testloader, device):
    tic = time.perf_counter()
    network.load_state_dict(torch.load(runpath+'model.pth'))
    preds, ids, fnames = None, None, None
    network.eval()
    with torch.no_grad():
        for i, (batch_images, batch_context, batch_ids, batch_fnames) in enumerate(testloader):
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

    submission = pd.DataFrame({'ID':ids, 'extent':preds})
    submission['extent'] = submission['extent'].clip(lower=0, upper=100)
    print(submission)
    submission.to_csv('pseudofile.csv', index=False)
    toc = time.perf_counter()
    print(f'pseudofile created in {round(toc-tic, 2)} seconds\n')

def create_assembly(path_extent, path_pred, runpath):
    df_extent = pd.read_csv(path_extent)
    df_pred = pd.read_csv(path_pred)
    df_pred.rename(columns={'extent':'pred'}, inplace=True)
    assembly = pd.concat([df_extent[['ID','filename','extent']], df_pred['pred']], axis=1)
    assembly['rpred'] = [round(el/5)*5 for el in assembly['pred']]
    assembly.to_csv(runpath+'assembly.csv', index=False)


def create_confusion_matrix(assembly, runpath):
    actuals = np.array(assembly['extent'].to_numpy())
    preds = np.array(assembly['rpred'].to_numpy())
    cmatrix = confusion_matrix(actuals, preds)
    cmatrix_df = pd.DataFrame(cmatrix, [i for i in range(5, 105, 5)], [i for i in range(5, 105, 5)])
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)
    sn.heatmap(cmatrix_df, annot=True, annot_kws={'size': 12}, cmap='Blues')
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.savefig(runpath+'confusion_matrix.png')
    plt.figure()
    plt.close('all')

