
import pandas as pd


class Splitter:
    def __init__(self, path_map, max=None):
        self.csv = pd.read_csv(path_map)
        self.trainsplit, self.valsplit = self.split()
        self.write()

    def split(self, ratio=0.5):
        self.csv = self.csv.sample(frac=1, random_state=1)
        wedge = int(len(self.csv)*ratio)
        trainsplit = self.csv[:wedge]
        valsplit = self.csv[wedge:]
        return trainsplit.sample(frac=1, random_state=1), valsplit.sample(frac=1, random_state=1)

    def write(self):
        self.trainsplit.to_csv('csv/set_0.csv', index=False)
        self.valsplit.to_csv('csv/set_1.csv', index=False)
        sample = pd.concat([self.trainsplit[:16], self.valsplit[:16]], axis=0)
        sample.to_csv('csv/sample.csv', index=False)
        print(len(self.trainsplit), len(self.valsplit))