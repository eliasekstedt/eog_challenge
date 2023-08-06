
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

def create_assembly(path_extent, path_pred, runpath):
    df_extent = pd.read_csv(path_extent)
    df_pred = pd.read_csv(path_pred)
    df_pred.rename(columns={'extent':'pred'}, inplace=True)
    assembly = pd.concat([df_extent[['ID','filename','extent']], df_pred['pred']], axis=1)
    assembly['rpred'] = [round(el/5)*5 for el in assembly['pred']]
    assembly.to_csv(runpath+'assembly.csv')


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

