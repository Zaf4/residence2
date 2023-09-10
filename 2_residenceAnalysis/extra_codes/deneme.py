import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from rich.progress import track
import matplotlib.colors as mcolors

#df
datafile = './data/durations_minimized.csv'
df = pd.read_csv(datafile, index_col=None)
#fit
fitfile = './data/PED.csv'
fits = pd.read_csv(fitfile, index_col=None)


fname = fitfile.split('/')[-1]
fname = fname[:-4]


df['timestep'] = np.arange(len(df))+1
fits['timestep'] = np.arange(len(df))+1

cols = [col for col in df]

fits = pd.melt(fits,
               id_vars=['timestep'],
               value_vars=cols,
               var_name='case',
               value_name='value')

df = pd.melt(df,
             id_vars=['timestep'],
             value_vars=cols,
             var_name='case',
             value_name='value')

# weights = ((1/(ts*ts[::-1]))*10**5)**2
ts = np.array(df.timestep)
tmax = df.timestep.max()
weights = np.abs(ts-tmax/2)
weights = 10**(weights/tmax)**3

#sampled data
sampled_df = df.sample(frac=0.02,random_state=42,weights=weights)
w = weights[:24000]

sns.lineplot(x=np.arange(len(w)),y=w)
plt.yscale('log')

















