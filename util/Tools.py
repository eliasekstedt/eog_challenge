
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import torch
import seaborn as sns

from datetime import datetime

def run_init(hparams, tag, device):
    # create directory for output files of current run
    current = datetime.now()
    runpath = f'run/{tag}/{str(current)[8:10]}_{str(current)[11:13]}_{str(current)[14:16]}_{str(current)[17:19]}/'
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

def show(image, runpath='', title=''):
    plt.imshow(image.to("cpu").permute(1, 2, 0))#, cmap='gray')
    #plt.title(title)
    plt.xticks([])
    plt.yticks([])
    #plt.savefig(runpath+title+'.png')
    #plt.figure()
    #plt.close('all')
    plt.show()


def performance_plot(model, runpath):
    epochs = range(1, len(model.testcost) + 1)
    traincol = 'tab:blue'
    testcol = 'tab:red'
    # figure
    fig, ax1 = plt.subplots(1,figsize=(6, 8))
    # cost
    ax1.plot(epochs, model.traincost, traincol, label='train')
    ax1.plot(epochs, model.testcost, testcol, label='val')
    ax1.set_ylim([4, 15])
    ax1.legend()
    #ax1.set_xticks([])
    ax1.set_ylabel('Cost')
    plt.tight_layout()
    plt.savefig(runpath+'performance'+'.png')
    plt.figure()
    plt.close('all')
    #plt.show()

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












