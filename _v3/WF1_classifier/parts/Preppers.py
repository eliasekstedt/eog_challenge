
import pandas as pd

outpath = 'WF1_classifier/csv/'

class Prep:
    def __init__(self):
        df = self.get_selection()
        df = self.hotncode(df)
        set_0, set_1 = self.split(df)
        print(set_0)
        print(set_1)
        set_0.to_csv(f'{outpath}set_0.csv', index=False)
        set_1.to_csv(f'{outpath}set_1.csv', index=False)

    def get_selection(self):
        df = pd.read_csv('csv/original/Train.csv').sample(frac=1)
        damage_yes = df.loc[df['extent'] >= 80]
        damage_no = df.loc[df['extent'] == 0].head(len(damage_yes))
        damage_yes['extent'] = [1]*len(damage_yes)
        damage_no['extent'] = [0]*len(damage_no)
        return pd.concat([damage_yes, damage_no], axis=0).sample(frac=1)

    def hotncode(sefl, df):
        categorical_columns = ['growth_stage', 'damage', 'season']
        return pd.get_dummies(df, columns=categorical_columns)
    
    def split(self, df):
        wedge = len(df)//2
        set_0 = df[0:wedge]
        set_1 = df[wedge:]
        return set_0, set_1

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

class HotnCode:
    def __init__(self, map_path, valmap_path):
        self.datamap = pd.read_csv(map_path)
        self.datamap = self.encoding(self.datamap)
        self.fold0, self.fold1 = self.split()
        self.valmap = pd.read_csv(valmap_path)
        self.valmap = self.encoding(self.valmap)
        self.valmap = self.valmap.sample(frac=1)
        self.write()
    
    def encoding(self, df):
        categorical_columns = ['growth_stage', 'damage', 'season']
        return pd.get_dummies(df, columns=categorical_columns)
        
    def split(self, ratio=0.5):
        self.datamap = self.datamap.sample(frac=1)
        wedge = int(len(self.datamap)*ratio)
        trainsplit = self.datamap[:wedge]
        testsplit = self.datamap[wedge:]
        return trainsplit, testsplit
    
    def write(self):
        self.fold0.to_csv('csv/fold_0.csv', index=False)
        self.fold1.to_csv('csv/fold_1.csv', index=False)
        self.valmap.to_csv('csv/Val.csv', index=False)

class Subset:
    def __init__(self, path, size):
        self.map0 = pd.read_csv(path['fold_0']).head(size)
        self.map1 = pd.read_csv(path['fold_1']).head(size)
        self.mapv = pd.read_csv(path['valmap']).head(size)
        self.map0.to_csv('csv/sub_0.csv', index=False)
        self.map1.to_csv('csv/sub_1.csv', index=False)
        self.mapv.to_csv('csv/sub_v.csv', index=False)




Prep()































