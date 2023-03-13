import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



#residual barplot 
df = pd.read_csv('./data/durations_minimized.csv',index_col=None)
fits = pd.read_csv('./data/powerlaw.csv',index_col=None)
df.index,fits.index = np.arange(len(df))+1, np.arange(len(df))+1

df.fillna(0)
fits.fillna(0)

ress = (df-fits).abs()


ress[ress==0] = np.nan


for um in ['60']:
    ax = plt.figure()
    cols = [col for col in ress if um in col]
    
    sns.set_palette('viridis')
    for col in cols:
        sns.lineplot(data = ress[col])
        plt.xscale('log')
        plt.yscale('log')
