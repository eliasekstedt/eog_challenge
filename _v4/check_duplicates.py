
import pandas as pd




set_0 = pd.read_csv('Workflow/csv/set_0.csv')
set_1 = pd.read_csv('Workflow/csv/set_1.csv')
val = pd.read_csv('Workflow/csv/val.csv')

#print(set_0)
#print(set_1)
#print(val)

#print(set_0.describe())
#print(set_1.describe())
#print(val.describe())

#print(set_0.info())
#print(set_1.info())
#print(val.info())
columns = set_0.columns
for col in columns:
    if col == 'extent':
        continue
    fnames_0 = set_0[col].tolist()
    fnames_1 = set_1[col].tolist()
    fnames_v = val[col].tolist()

    get_n_duplicates = lambda lst0, lst1: sum([(fname in lst1)*1 for fname in lst0])
    s0 = get_n_duplicates(fnames_0, fnames_1)
    s1 = get_n_duplicates(fnames_1, fnames_v)
    v = get_n_duplicates(fnames_v, fnames_0)
    print(f'{col}: {(s0, s1, v)}')




"""
strange behaviour should always be addressed directly
"""