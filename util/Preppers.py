
import pandas as pd

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

map_path = 'csv/original/Train.csv'
valmap_path = 'csv/original/Test.csv'
HotnCode(map_path, valmap_path)































