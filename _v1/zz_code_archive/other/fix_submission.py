
import numpy as np
import pandas as pd



def fix1():
    runpath = 'run/27_16_09_03/'
    df = pd.read_csv(runpath + 'SampleSubmission.csv')
    df['extent'] = df['extent'].clip(lower=0, upper=100)
    ss_round, ss_floor, ss_ceil = df.copy(), df.copy(), df.copy(),
    ss_round['extent'] = (ss_round['extent']/10).round()*10
    ss_floor['extent'] = np.floor(ss_floor['extent']/10)*10
    ss_ceil['extent'] = np.ceil(ss_ceil['extent']/10)*10

    print(df)
    print(ss_round)
    print(ss_floor)
    print(ss_ceil)
    df.to_csv('oSampleSubmission.csv', index=False)
    ss_round.to_csv('rSampleSubmission.csv', index=False)
    ss_floor.to_csv('fSampleSubmission.csv', index=False)
    ss_ceil.to_csv('cSampleSubmission.csv', index=False)

    #common = df.merge(df2, on='ID')['ID'].tolist()
    #print(f'df    : {len(df)}')
    #print(f'df2   : {len(df2)}')
    #print(f'common: {len(common)}')

def fix2():
    runpath = 'run/main/01_09_51_24/'
    df = pd.read_csv(runpath + '200_15_res18fc_1e-07_0.0_(128, 128).csv')
    extent0 = [0]*len(df)
    extent10 = [10]*len(df)
    extent50 = [50]*len(df)
    extent90 = [90]*len(df)
    extent100 = [100]*len(df)
    df0 = df.copy()
    df10 = df.copy()
    df50 = df.copy()
    df90 = df.copy()
    df100 = df.copy()
    
    df0['extent'] = extent0
    df10['extent'] = extent10
    df50['extent'] = extent50
    df90['extent'] = extent90
    df100['extent'] = extent100
    print(df0)
    print(df10)
    print(df50)
    print(df90)
    print(df100)
    f_ind = [0, 10, 50, 90, 100]
    for i, dframe in enumerate([df0, df10, df50, df90, df100]):
        dframe.to_csv(f'sub_{f_ind[i]}.csv', index=False)


def main():
    fix2()
    #fix1()

if __name__ == '__main__':
    main()