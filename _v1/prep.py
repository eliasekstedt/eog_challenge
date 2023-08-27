

hotncode = False
sub = True

if hotncode:
    from util.Preppers import HotnCode as Prep
    path = {'map':'csv/original/Train.csv',
            'val':'csv/original/Test.csv'}
    Prep(path['map'], path['val'])

if sub:
    from util.Preppers import Subset
    path = {'fold_0':'csv/fold_0.csv',
            'fold_1':'csv/fold_1.csv',
            'valmap':'csv/Val.csv',
            }
    Subset(path, 500)