
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def scatter_matrix(df, metric):
    sns.set(style="ticks")
    sns.pairplot(df)
    plt.savefig(f'scatter_matrix_{metric}.png')
    plt.figure()
    plt.close('all')
    #plt.show()

def main():
    path = {'labeled':'csv/original/Train.csv'}

    data = pd.read_csv(path['labeled'])
    data = data[data.columns[2:]]
    print(data)
    data = data.loc[data['extent']>=90]
    print(data)

    avg_extent = data
    #scatter_matrix(data, 'extent')

if __name__ == '__main__':
    main()






















#data = data.drop(data.columns[data.columns.str.startswith('damage')], axis=1)

