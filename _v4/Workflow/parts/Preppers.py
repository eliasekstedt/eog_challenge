
import pandas as pd

class HotnCode:
    def __init__(self, csv_path, val_path, out_dir, subset):
        self.out_dir = out_dir
        df = pd.read_csv(csv_path).sample(frac=1)
        df = self.balance_sub_select(df)
        df = self.encoding(df)
        set_0, set_1 = self.split(df)
        val = pd.read_csv(val_path).sample(frac=1)
        val = self.encoding(val)
        val = val.sample(frac=1)
        print(f'set_sizes:\nset_0:\t{len(set_0)}\nset_1:\t{len(set_1)}\nval:\t{len(val)}')
        self.write(set_0, set_1, val, out_dir)
    
    def balance_sub_select(self, df):
        extent_gtz = df.loc[df['extent'] > 0]
        extent_z = df.loc[df['extent'] == 0].head(len(extent_gtz)//10)
        df = pd.concat([extent_gtz, extent_z], axis=0).sample(frac=1)
        return df

    def encoding(self, df):
        categorical_columns = ['growth_stage', 'damage', 'season']
        df = pd.get_dummies(df, columns=categorical_columns)
        return df
        
    def split(self, df, ratio=0.5):
        wedge = int(len(df)*ratio)
        set_0 = df[:wedge]
        set_1 = df[wedge:]
        return set_0, set_1
    
    def write(self, set_0, set_1, val, out_dir):
        set_0.to_csv(out_dir+'set_0.csv', index=False)
        set_1.to_csv(out_dir+'set_1.csv', index=False)
        val.to_csv(out_dir+'val.csv', index=False)

map_path = 'CSV/original/Train.csv'
valmap_path = 'CSV/original/Test.csv'
out_dir = 'Workflow/csv/'

HotnCode(map_path, valmap_path, out_dir, subset=False)


"""
outpath = 'Workflow/csv/'
# classification
class Prep:
    def __init__(self):
        df = self.get_selection()
        df = self.hotncode(df)
        set_0, set_1, val = self.split(df)
        print(f'lens: {len(set_0)}, {len(set_1)}, {len(val)}')
        set_0.to_csv(f'{outpath}set_0.csv', index=False)
        set_1.to_csv(f'{outpath}set_1.csv', index=False)
        val.to_csv(f'{outpath}val.csv', index=False)

    def get_selection(self):
        df = pd.read_csv('csv/original/Train.csv').sample(frac=1)
        damage_yes = df.loc[df['extent'] >= 60]
        damage_no = df.loc[df['extent'] == 0].head(len(damage_yes))
        damage_yes['extent'] = [1]*len(damage_yes)
        damage_no['extent'] = [0]*len(damage_no)
        return pd.concat([damage_yes, damage_no], axis=0).sample(frac=1)

    def hotncode(self, df):
        categorical_columns = ['growth_stage', 'damage', 'season']
        return pd.get_dummies(df, columns=categorical_columns)
    
    def split(self, df):
        wedge0 = len(df)//2
        val = df[:wedge0]
        train = df[wedge0:]
        wedge = len(train)//2
        set_0 = train[:wedge]
        set_1 = train[wedge:]
        return set_0, set_1, val

class Basic:
    def __init__(self, map_path):
        self.datamap = pd.read_csv(map_path)
        self.trainsplit, self.testsplit = self.split()
        self.write()

    def split(self, ratio=0.5):
        self.datamap = self.datamap.sample(frac=1)
        wedge = int(len(self.datamap)*ratio)
        trainsplit = self.datamap[:wedge]
        testsplit = self.datamap[wedge:]
        return trainsplit, testsplit
    
    def write(self):
        self.trainsplit.to_csv('csv/trainsplit.csv', index=False)
        self.testsplit.to_csv('csv/testsplit.csv', index=False)
        
class Overview:
    def __init__(self, map_path, valmap_path):
        self.datamap = pd.read_csv(map_path)
        self.datamap = self.datamap.loc[self.datamap['extent'] == 0]
        vcount_dam = self.datamap['damage'].value_counts()
        vcount_gs = self.datamap['growth_stage'].value_counts()
        vcount_s = self.datamap['season'].value_counts()
        print(self.datamap)
        print(vcount_dam)
        print(vcount_gs)
        print(vcount_s)



class Subset:
    def __init__(self, path, size):
        self.map0 = pd.read_csv(path['set_0']).head(size)
        self.map1 = pd.read_csv(path['set_1']).head(size)
        self.mapv = pd.read_csv(path['valmap']).head(size)
        self.map0.to_csv('csv/sub_0.csv', index=False)
        self.map1.to_csv('csv/sub_1.csv', index=False)
        self.mapv.to_csv('csv/sub_v.csv', index=False)

"""


































