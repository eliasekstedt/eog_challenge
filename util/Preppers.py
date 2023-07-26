
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




class SimpleSet:
    def __init_(self, map_path):
        self.datamap = pd.read_csv(map_path)
    
    def select_simple(self):
        print(self.datamap)
































