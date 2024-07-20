
import os
from datetime import datetime

def file_it(file_name, message, to_terminal=False):
    if to_terminal:
        print(message)
    with open(file_name, 'a') as file:
        file.write(f'{message}\n')

def run_init(hparam, augmentation, tag, device):
    current = datetime.now()
    runpath = f'run/{tag}/{str(current)[8:10]}_{str(current)[11:13]}_{str(current)[14:16]}_{str(current)[17:19]}/'
    # create dir for run history if none exists
    if not os.path.exists('run/history/'):
        os.makedirs('run/history/')
    # create directory for output files of current run
    if not os.path.isdir(runpath):
        os.makedirs(runpath)
    # saving runlog to current run folder
    with open(os.path.join(runpath, 'log.txt'), 'a') as file:
        file.write('################# initiating run #################\n')
        file.write(f'run dir has been created in {runpath}\n\n')
        for param in hparam:
            file.write(f'{param}\t: {hparam[param]}\n')
        for aug in augmentation:
            file.write(f'{aug}\t: {augmentation[aug]}\n')
        file.write('\n')
        file.write(f'Using {device} device\n')
        file.write('################# initiated run ##################\n')
    # printing current state of logfile to terminal
    with open(runpath+'log.txt', 'r') as file:
        for_terminal = file.read()
    print(for_terminal)
    return runpath


