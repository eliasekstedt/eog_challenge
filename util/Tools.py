
import matplotlib.pyplot as plt
import os

from datetime import datetime

def run_init(hparams, device):
    # create directory for output files of current run
    current = datetime.now()
    runpath = f'run/{str(current)[8:10]}_{str(current)[11:13]}_{str(current)[14:16]}_{str(current)[17:19]}/'
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
        file.write('################# initiated run ##################')
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
    epochs = range(1, len(model.tune_cost) + 1)
    tunecol = 'tab:blue'
    valcol = 'tab:red'
    # figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    # cost
    ax1.plot(epochs, model.tune_cost, tunecol, label='train')
    ax1.plot(epochs, model.val_cost, valcol, label='val')
    ax1.legend()
    ax1.set_xticks([])
    ax1.set_ylabel('Cost')
    # accuracy
    ax2.plot(epochs, model.tune_accuracy, tunecol, label='train')
    ax2.plot(epochs, model.val_accuracy, valcol, label='val')
    ax2.legend()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(runpath+'performance'+'.png')
    plt.figure()
    plt.close('all')
    #plt.show()
















