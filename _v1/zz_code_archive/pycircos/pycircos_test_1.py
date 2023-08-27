
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycircos
Garc = pycircos.Garc
Gcircle = pycircos.Gcircle


class LabelPredMap:
    def __init__(self, path):
        self.df = pd.read_csv(path['data'])
        self.circle = Gcircle(figsize=(8,8))
        self.place_foundation()
        self.place_ticks()
        self.place_links()
        self.plot(save_as=path['figure_name'])

    def place_foundation(self):
        id = 'chr'
        arc = Garc(arc_id=id, size=100, interspace=1, linewidth=2, facecolor='#FFFFFF00', raxis_range=(920,930), label_visible=False)
        self.circle.add_garc(arc)
        self.circle.set_garcs(-65, 115)
        #self.circle.set_garcs()

    def place_ticks(self):
        self.circle.tickplot('chr', raxis_range=(930,950), tickinterval=10, ticklabels=None)#list(range(100))[::10])

    def place_links(self):
        def get_shade(intensity, reach):
            if reach >= 20:
                shade = '#2AB4F5'
            else:
                shades = ['#000000FF', '#1A1A1AFF', '#343434FF', '#4E4E4EFF', '#686868FF', '#828282FF', '#9C9C9CFF', '#B6B6B6FF', '#D0D0D0FF', '#FFFAFAFA']
                shade = shades[1] # shades[int(round(intensity/10))]
            return shade

        for i in range(len(self.df)):
            id = 'chr'
            pos_ori = self.df['pos_label'][i]
            pos_dest = self.df['pos_pred'][i]
            reach = np.abs(pos_dest - pos_ori)
            if reach > 10:
                source = (id, pos_ori - 0.02, pos_ori, 930)
                target = (id, pos_dest - 0.02, pos_dest, 930)
                self.circle.chord_plot(source, target, facecolor=get_shade(pos_ori, reach))

    def plot(self, save_as):
        self.circle.figure
        plt.savefig(f'zz_code_archive/pycircos/{save_as}')
        plt.figure()
        plt.close('all')
        plt.show()

def main():
    path = {'data':'zz_code_archive/pycircos/pred_label_data.csv',
            'figure_name':'circle_1'}
    
    LabelPredMap(path)

if __name__ == '__main__':
    main()