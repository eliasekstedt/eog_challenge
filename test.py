
import pandas as pd



def test0():
    def get_subfile_name(runpath, hparam):
        subname_components = runpath
        subname_components = [runpath]+[str(param)+'_' for param in hparam.values()]
        subname = ''
        for i in range(len(subname_components)):
            subname = subname + subname_components[i]
        return subname[:-1] + '.csv'


    hparam = {'batch_size': 200,
            'nr_epochs': 15,
            'architecture_name':'res18fc',
            'weight_decay': 1e-7,
            'dropout_rate': 0.0,
            'resizes':(128, 128)}

    runpath = '' #'run/00_00_00_00/'
    subname = get_subfile_name(runpath, hparam)

    df = pd.read_csv('allsizes.csv')

    df.to_csv(subname, index=False)


def main():
    test0()

if __name__ == '__main__':
    main()