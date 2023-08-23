
import torch
from torch.utils.data import DataLoader
import pandas as pd

def create_submission(runpath, model, loader, hparam, device):
    print('creating submission file')
    def get_subfile_name(runpath, hparam):
        subname_components = runpath
        subname_components = [runpath]+[str(param)+'_' for param in hparam.values()]
        subname = ''
        for i in range(len(subname_components)):
            subname = subname + subname_components[i]
        return subname[:-1] + '.csv'
    
    preds, ids, fnames = None, None, None
    model.eval()
    with torch.no_grad():
        for batch_images, batch_context, batch_ids, batch_fnames in loader:
            batch_images = batch_images.to(device)
            batch_context = batch_context.to(device)
            batch_outputs = model(batch_images, batch_context)
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
    print(f'sub-file created {runpath}')


def main():
    device = 'cuda:0'
    path = {'fold_0':'csv/trainsplit.csv',
            'fold_1':'csv/testsplit.csv',
            'valmap':'csv/Val.csv',
            'data_labeled':'data/train/',
            'data_unlabeled':'data/test/'
            }
    # hyperparameters
    hparam = {'batch_size': 100,
            'nr_epochs': 25,
            'architecture_name':'nex',
            'weight_decay': 1e-7,
            'dropout_rate': 0.0,
            'resizes':(128, 128),
            'penalty': 1}

    runpath = 'run/23_14_02_21_special/'
    model_name = 'model_0.pth'

    # loading data
    from util.Readers import Res18FCReader as reader
    set = reader(path['valmap'], path['data_unlabeled'], resizes=hparam['resizes'], augment=False, eval=True)
    loader = DataLoader(set, batch_size=hparam['batch_size'], shuffle=False)

    # load model
    from util.Networks import Res18FCNet
    model = Res18FCNet(0, hparam['architecture_name'], hparam['weight_decay'], hparam['dropout_rate'], hparam['penalty']).to(device)
    model.load_state_dict(torch.load(runpath+model_name))
    model.eval()

    # create submission file
    create_submission(runpath, model, loader, hparam, device)


if __name__ == '__main__':
    main()