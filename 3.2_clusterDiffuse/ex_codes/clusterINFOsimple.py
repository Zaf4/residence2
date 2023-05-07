import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def simplify(df:pd.DataFrame)->pd.DataFrame:
    
    df = df[['clusterID','residence','clusterSize','timestep','conformation','kT']]
    df = df.drop_duplicates(['clusterID','timestep'])
    df = df[df.clusterID>0]
    df = df[df.residence>0]
    return df


def merge_dfs(df_files:list)->pd.DataFrame:


    df_all = pd.DataFrame()

    for file in df_files:

        df = pd.read_csv(f'./data/40timepoint/{file}.csv',index_col=None)
        df = simplify(df)
        df_all = pd.concat([df_all,df])

    df_all.index = np.arange(len(df_all))
    
    return df_all


if __name__ == '__main__':
    files = [280,300,350,400]
    df_all = merge_dfs(files)

    sns.boxplot(data=df_all,x='conformation',y='residence',
                hue='kT',)
    plt.yscale('log')

    sns.lineplot(data=df_all,x='kT',y='residence',
                hue='conformation',)
    plt.yscale('log')