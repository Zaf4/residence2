import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statannot
import os

#changing working directory to current directory name
os.chdir(os.path.dirname(__file__))

def round_up(value:float|int):
    n_digits = len(str(int(value)))
    n = n_digits-2
    if value<9:
        value+=1
    upper = np.ceil((value)/10**n)*10**n
    return upper

#importing dataframes
df280 = pd.read_csv('./data/12timepoint/280.csv', index_col=False)
df300 = pd.read_csv('./data/12timepoint/300.csv', index_col=False)
df350 = pd.read_csv('./data/12timepoint/350.csv', index_col=False)
df400 = pd.read_csv('./data/12timepoint/400.csv', index_col=False)

dfs = [df280,df300,df350,df400]

#mapping dict for preprocessing
mapping = {
    0: 'free',
    1: 'surf',
    2: 'core'
}

#font parameters
fontparams = {         
              'fontweight':'light',
              'fontsize':20,
              'fontname':'Sans Serif',

              }

#setting of graphs
sns.set_theme(style='ticks',
    rc = {
          'font.weight':'light',
          'font.size':20,
          'font.family':'Sans Serif',
          'ytick.minor.size':'0',
          'ytick.major.size':'10',
          'xtick.major.size':'10'
          
          }
    )

#Initing multiplot
fig, axes = plt.subplots(2,4,figsize=(20,8),)
fig.subplots_adjust(wspace=0.2, hspace=0.4)

for i,df in enumerate(dfs):
    
    #0,1,2 replaced to free,surf,core kinds
    df['kind'] = df['kind'].replace(mapping)
    
    #taking only one bead for a TF
    df = df[df.type==5]
    
    
    #names for labeling
    kts = [2.8, 3, 3.5, 4]

    #Barplot-------------------------------------------------------------------
    sns.barplot(data=df,
                  x='kind',y='residence',
                  ax=axes[0,i], order = ['free','surf','core'],
                  palette='husl',edgecolor='k',saturation=1,
                  errcolor='k', errwidth=1.5, capsize=0.3)
    

    #Swarmplot-----------------------------------------------------------------

    swarm_data = df.sample(frac=0.12,random_state = 42,weights=df.residence**2)
    sns.swarmplot(data=swarm_data,
                  x='kind',y='residence',
                  ax=axes[1,i],size=2,order = ['free','surf','core'],
                  palette='husl',edgecolor='k')
    

    #p-values
    stat1,pvalue1 = stats.ttest_ind(df[df.kind==0], df[df.kind==1])
    stat2,pvalue2 = stats.ttest_ind(df[df.kind==1], df[df.kind==2])
    stat3,pvalue3 = stats.ttest_ind(df[df.kind==0], df[df.kind==2])
    
    #adding significance annotations -test: Welch's t-test---------------------
    statannot.add_stat_annotation(axes[0,i],
                                  data=df,
                                  x='kind',
                                  y='residence',
                                  order = ['free','surf','core'],
                                  # hue=None,
                                  box_pairs=[('free','surf'),
                                             ('surf','core'), 
                                             ('free','core')
                                             ],
                                  test="t-test_welch",
                                  text_format='star',
                                  loc="outside",
                                  width=1,
                                  color='0.4',
                                  line_height=0.01,
                                  text_offset=0.1,
                                  fontsize='small')

    
    #graph settings------------------------------------------------------------
    for j in range(2):
        ax = axes[j,i]
        ax.set_xlabel(None)
        ax.set_ylabel('Residence Time (a.u.)',**fontparams)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Free', 'Surface', 'Core'],
                            fontsize=22,fontweight='light')
        
        ax.tick_params(axis='y',
                       labelsize=18)
    
        ax.set_title(r'$U_{ns} = $' + f'{kts[i]}kT',
                     fontsize=20,
                     style='italic',
                     fontweight='light',
                     fontname='Arial')
        
        if j==0:
            lower,upper = ax.get_ylim()
            ax.set_ylim([lower,upper*1.5])

#overall figure setting and saving
sns.despine(trim=True)

plt.tight_layout()

# plt.annotate('A',xycoords='figure fraction', xy = (0.01,0.93),fontsize=48)
# plt.annotate('B',xycoords='figure fraction', xy = (0.01,0.45),fontsize=48)


fig.savefig('../Figures/fig5.pdf',transparent=True)
fig.savefig('../Figures/fig5.png',transparent=True,dpi=300)