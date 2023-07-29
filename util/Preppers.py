
import pandas as pd

class Basic:
    def __init__(self, map_path):
        self.datamap = pd.read_csv(map_path)
        self.trainsplit, self.testsplit = self.split()
        self.write()
        #print(self.trainsplit)
        #print(self.testsplit)


    def split(self, ratio=0.5):
        self.datamap = self.datamap.sample(frac=1)
        wedge = int(len(self.datamap)*ratio)
        trainsplit = self.datamap[:wedge]
        testsplit = self.datamap[wedge:]
        return trainsplit, testsplit
    
    def write(self):
        self.trainsplit.to_csv('csv/trainsplit.csv', index=False)
        self.testsplit.to_csv('csv/testsplit.csv', index=False)
        

"""
    def get_categories(self):
        #df = self.datamap.groupby(['growth_stage', 'damage', 'season'])
        df = self.datamap.groupby(['growth_stage'])
        #df = self.datamap.groupby(['damage'])
        #df = self.datamap.groupby(['season'])
        categories = {}
        for category, group in df:
            #category_name = category[0]+'_'+category[1]+'_'+category[2]
            category_name = category
            #print(category_name, len(group))
            categories.update({category_name: group})
        print(categories.keys())
        print(categories.values())
        return categories
            
"""

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
        self.trainsplit, self.testsplit = self.split()
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
        self.trainsplit.to_csv('csv/trainsplit.csv', index=False)
        self.testsplit.to_csv('csv/testsplit.csv', index=False)
        self.valmap.to_csv('csv/Val.csv', index=False)

map_path = 'csv/original/Train.csv'
valmap_path = 'csv/original/Test.csv'
HotnCode(map_path, valmap_path)































