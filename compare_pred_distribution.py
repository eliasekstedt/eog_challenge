
import matplotlib.pyplot as plt
import pandas as pd







def main():
    path = {'to_df1': 'csv/original/Train.csv',
            'to_df2': '256_15_res18fc_1e-07_0.0__128__128_.csv',
            'model': 'current.pth'
            }
    

    if False:
        df1 = pd.read_csv(path['to_df1'])
        df2 = pd.read_csv(path['to_df2'])

        dist1 = list(df1.loc[df1['extent']>=0]['extent'])
        dist2 = list(df2.loc[df2['extent']>=0]['extent'])

        dist1 = [round(val/10)*10 for val in dist1]
        dist2 = [round(val/10)*10 for val in dist2]

        print(dist1)
        print(dist2)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.hist(dist1, bins='auto')
        ax2.hist(dist2, bins='auto')
        plt.show()


if __name__ == '__main__':
    main()