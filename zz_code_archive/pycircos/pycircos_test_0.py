
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycircos
import collections
Garc = pycircos.Garc
Gcircle = pycircos.Gcircle

class PredResultsPlot:
    def __init__(self, path_csv, use, save_as='circle'):
        self.labelreducer = int(1e1)
        self.predreducer = int(3e1)
        self.df = pd.read_csv(path_csv)
        print(self.df)
        self.circle = Gcircle(figsize=(8,8))
        self.place_foundation()
        if use['ticks']:
            self.place_ticks()
        if use['bands']:
            self.banddata = collections.defaultdict(dict)
            self.place_bands()
        if use['links']:
            self.linkdata = collections.defaultdict(dict)
            self.place_links()
        self.plot(use, save_as)

    def place_foundation(self):
        label_id = 'label'
        pred_id = 'pred'
        #length = 100*self.linkreducer
        #label_arc = Garc(arc_id=label_id, size=length, interspace=2, raxis_range=(935, 985), labelposition=80, label_visible=True)
        #pred_arc = Garc(arc_id=pred_id, size=length, interspace=2, raxis_range=(935, 985), labelposition=80, label_visible=True)
        label_arc = Garc(arc_id=label_id, size=100*self.labelreducer, interspace=2, linewidth=2, facecolor='#FFFFFF00', raxis_range=(880,890), labelposition=80, label_visible=True)
        pred_arc = Garc(arc_id=pred_id, size=100*self.predreducer, interspace=2, linewidth=2, facecolor="#FFFFFF00", raxis_range=(980,990), labelposition=80, label_visible=True)
        self.circle.add_garc(label_arc)
        self.circle.add_garc(pred_arc)
        self.circle.set_garcs()


    def place_ticks(self):
        #for arc_id in self.circle.garc_dict:
        self.circle.tickplot('label', raxis_range=(890,900), tickinterval=self.labelreducer, ticklabels=None)
        self.circle.tickplot('pred', raxis_range=(990,1000), tickinterval=self.predreducer, ticklabels=None)

    def place_bands(self):
        color_dict = {'gneg':'#FFFFFF00', 'gpos25':'#EEEEEE', 'gpos50':'#BBBBBB', 'gpos75':'#777777', 'gpos100':'#000000', 'gvar':'#FFFFFF00', 'stalk':'#C01E27', 'acen':'#D82322'}
        for row in range(len(self.df)):
            id = self.df['id'][row]
            start = self.df['band_start_pos'][row]
            width = self.df['band_end_pos'][row] - (self.df['band_start_pos'][row])
            if id not in self.banddata:
                self.banddata[id]['begin_pos'], self.banddata[id]['widths'], self.banddata[id]['colors'] = [], [], []
            self.banddata[id]['begin_pos'].append(start)
            self.banddata[id]['widths'].append(width)
            color = 'gpos100' #list(color_dict.keys())[np.random.randint(0, len(color_dict))]
            self.banddata[id]['colors'].append(color_dict[color])
        
    def place_links(self):
        def get_shade(intensity):
            shades = ['#000000FF', '#1A1A1AFF', '#343434FF', '#4E4E4EFF', '#686868FF', '#828282FF', '#9C9C9CFF', '#B6B6B6FF', '#D0D0D0FF', '#FFFAFAFA']
            return shades[int(round(intensity/(10*self.labelreducer)))]

        for i in range(len(self.df)):
            id_ori = 'label'
            id_dest = 'pred'
            pos_ori = self.df['pos_label'][i]*self.labelreducer
            pos_dest = self.df['pos_pred'][i]*self.predreducer
            source = (id_ori, pos_ori - 1, pos_ori, 880)
            target = (id_dest, pos_dest -1, pos_dest, 980)
            self.circle.chord_plot(source, target, facecolor=get_shade(pos_ori))

    def plot(self, use, save_as):
        if use['bands']:
            for key in self.banddata:
                print(f'key: {key}, value: {self.banddata[key]}')
                self.circle.barplot(key, data=[1]*len(self.banddata[key]['begin_pos']), positions=self.banddata[key]['begin_pos'],
                                    width=self.banddata[key]['widths'], raxis_range=[935, 985], facecolor=self.banddata[key]['colors'])                
        self.circle.figure
        if not save_as == '':
            plt.savefig(f'zz_code_archive/pycircos/{save_as}')
            plt.figure()
            plt.close('all')
        else:
            plt.show()

class CircosPrep:
    def __init__(self, path):
        df = pd.read_csv(path['csv_label'])
        df['id'] = 'label'
        df.rename(columns={'extent':'pos_label'}, inplace=True)
        df_pred = pd.read_csv(path['csv_pred'])
        df['pos_pred'] = df_pred['extent']
        self.df = df[['pos_label', 'pos_pred']]
        self.write()

    def write(self):
        self.df.to_csv('zz_code_archive/pycircos/pred_label_data.csv', index=False)

def main():
    #print(df.describe())

    #df = pd.read_csv('csv/trainsplit.csv')
    #df['pseudo'] = np.abs([np.random.normal(m, 3) for m in df['extent']])

    do_prep = False
    do_prep = False
    #do_prep = True
    do_plot = True
        
    path = {'csv':'zz_code_archive/pycircos/applied_chr_data.csv',
            'csv_label':'zz_code_archive/pycircos/256_15_res18fc_1e-07_0.0__128__128_.csv',
            'csv_pred':'zz_code_archive/pycircos/128_20_res18fc_1e-07_0.0_(256, 256).csv',
            'csv_circos_data':'zz_code_archive/pycircos/pred_label_data.csv', 
            'figure_name': 'circle_0'}


    
    if do_prep:
        CircosPrep(path)


    if do_plot:
        use = {'ticks':True,
                'bands':False,
                'links':True}

        PredResultsPlot(path['csv_circos_data'], use, save_as=path['figure_name'])







if __name__ == '__main__':
    main()












"""
##################################
chr0: predictions extent value
chr1: labels extent value

ticks0: extent value
ticks1: extent value

links: pred extent to label extent for each datapoint
##################################
what i want to show / what information to include:
* label -> prediction in extent value
    link from extent value on label chr to extent value on pred chr
    implication:
        * ticks define extent value
        * each datapoint its own prediction so
* 




chr definition
* one chr for prediction and label each
tick definition
* value in extent column
    
"""







