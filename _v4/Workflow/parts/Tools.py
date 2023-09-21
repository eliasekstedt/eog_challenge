
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import torch


def run_init(hparams, tag, device):
    current = datetime.now()
    runpath = f'run/{tag}/{str(current)[8:10]}_{str(current)[11:13]}_{str(current)[14:16]}_{str(current)[17:19]}/'
    # saving runpath in run history
    if not os.path.isdir('run/_history'):
        os.makedirs('run/_history')
    with open('run/_history/history.txt', 'a') as file:
        file.write(f'\n{runpath}')
    # create directory for output files of current run
    if not os.path.isdir(runpath):
        os.makedirs(runpath)
    # saving runlog to current run folder
    with open(os.path.join(runpath, 'log.txt'), 'a') as file:
        file.write('################# initiating run #################\n')
        file.write(f'run dir has been created in {runpath}\n\n')
        for param in hparams:
            file.write(f'{param}\t: {hparams[param]}\n')
        file.write('\n')
        file.write(f'Using {device} device\n')
        file.write('################# initiated run ##################\n')
    # printing current state of logfile to terminal
    with open(runpath+'log.txt', 'r') as file:
        for_terminal = file.read()
    print(for_terminal)
    return runpath

def to_log(runpath, messages):
    with open(runpath+'log.txt', 'a') as file:
        for message in messages:
            file.write(message+'\n')


def show(image, runpath='', title=''):
    plt.imshow(image.to("cpu").permute(1, 2, 0))#, cmap='gray')
    #plt.title(title)
    plt.xticks([])
    plt.yticks([])
    #plt.savefig(runpath+title+'.png')
    #plt.figure()
    #plt.close('all')
    plt.show()

def plot_performance(model, runpath):
    epochs = range(1, len(model.testcost) + 1)
    traincol = 'tab:blue'
    testcol = 'tab:red'
    # figure
    fig, (ax1, ax2) = plt.subplots(2,figsize=(6, 8))
    # cost
    ax1.plot(epochs, model.traincost, traincol, label='train')
    ax1.plot(epochs, model.testcost, testcol, label='val')
    ax1.set_ylim([0, 20])
    ax1.legend()
    #ax1.set_xticks([])
    ax1.set_ylabel('Cost')
    ax2.plot(epochs, model.s_traincost, traincol, label='train')
    ax2.plot(epochs, model.s_testcost, testcol, label='val')
    ax2.set_ylim([0, 20])
    ax2.legend()
    ax2.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(f'{runpath}performance.png')
    plt.figure()
    plt.close('all')

def plot_by_ctx_feature(runpath):
    """for each of the four categories, the mean of every feature"""
    df = pd.read_csv(f'{runpath}evaldata.csv')
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
    rn_dict = {'growth_stage_F':'gsf', 'growth_stage_M':'gsm', 'growth_stage_S':'gss', 'growth_stage_V':'gsv',
       'damage_DR':'ddr', 'damage_DS':'dds', 'damage_FD':'dfd', 'damage_G':'dg', 'damage_ND':'dnd', 'damage_PS':'dps',
       'damage_WD':'dwd', 'damage_WN':'dwn','season_LR2020':'sl0', 'season_LR2021':'sl1', 'season_SR2020':'ss0',
       'season_SR2021':'ss1'}
    df.rename(columns=rn_dict, inplace=True)
    categories = df.columns[1:]
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
    plt.savefig(f'{runpath}bars')
    plt.figure()
    plt.close('all')



def create_submission(network, runpath, valloader, hparam, device):
    def get_subfile_name(runpath, hparam):
        subname_components = runpath
        subname_components = [runpath]+[str(param)+'_' for param in hparam.values()]
        subname = ''
        for i in range(len(subname_components)):
            subname = subname + subname_components[i]
        return subname[:-1] + '.csv'
    tic = time.perf_counter()
    network.load_state_dict(torch.load(runpath+'model.pth'))
    preds, ids, fnames = None, None, None
    network.eval()
    with torch.no_grad():
        for i, (batch_images, batch_context, batch_ids, batch_fnames) in enumerate(valloader):
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
    subfile_name = get_subfile_name(runpath, hparam)
    submission.to_csv(subfile_name, index=False)
    toc = time.perf_counter()
    print(f'sub-file created in {round(toc-tic, 2)} seconds\n')

def scatter_matrix(df, metric):
    sns.set(style="ticks")
    sns.pairplot(df)
    plt.savefig(f'scatter_matrix_{metric}.png')
    plt.figure()
    plt.close('all')
    #plt.show()

def gp_plot(df, setup, logpath):
    xlim = setup['bounds']
    param = setup['key_for_opt'][0]
    x = np.array(df[param].to_list())
    y = np.array([[yy] for yy in df['unclipped']])
    plt.plot(x, y, 'o', color='black')
    plt.plot(x[-1], y[-1], 'o', color='blue')
    plt.xlim(xlim)
    plt.savefig(f'{logpath}scatter.png')
    plt.figure()
    plt.close('all')









