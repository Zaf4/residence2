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
              'fontsize':22,
              'fontname':'Sans Serif',

              }

#setting of graphs
sns.set_theme(style='ticks',
    rc = {
          'font.weight':'light',
          'font.size':18,
          'font.family':'Sans Serif',
          'ytick.minor.size':'0',
          'ytick.major.size':'10',
          'xtick.major.size':'10'
          
          }
    )

#Initing multiplot
fig, axes = plt.subplots(4,2,figsize=(15,18))
fig.subplots_adjust(wspace=0.15, hspace=0.15)

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
                  ax=axes[i,0], order = ['free','surf','core'],
                  palette='husl',edgecolor='k',saturation=1,
                  errcolor='k', errwidth=1.5, capsize=0.3)
    

    #tf ratios
    N = len(df)
    ratio_surf = round(len(df[df.kind=='surf'])/N*100)
    ratio_free = round(len(df[df.kind=='free'])/N*100)
    ratio_core = round(len(df[df.kind=='core'])/N*100)

    #Swarmplot-----------------------------------------------------------------
    # 0.12
    swarm_data = df.sample(frac=0.12,random_state = 42,weights=df.residence**2)
    sns.swarmplot(data=swarm_data,
                  x='kind',y='residence',
                  ax=axes[i,1],size=2,order = ['free','surf','core'],
                  palette='husl',edgecolor='k')
    

    #p-values
    stat1,pvalue1 = stats.ttest_ind(df[df.kind==0], df[df.kind==1])
    stat2,pvalue2 = stats.ttest_ind(df[df.kind==1], df[df.kind==2])
    stat3,pvalue3 = stats.ttest_ind(df[df.kind==0], df[df.kind==2])
    
    #adding significance annotations -test: Welch's t-test---------------------
    statannot.add_stat_annotation(axes[i,0],
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
                                  line_height=0.04,
                                  text_offset=0,
                                  fontsize='small',
                                  verbose=0)

    
    #graph settings------------------------------------------------------------
    for j in range(2):
        ax = axes[i,j]
        ax.set_xlabel(None)
        # if j == 0:
        #     ax.set_ylabel('Residence Time (a.u.)',**fontparams)
        # else:
        #     ax.set_ylabel('')
        ax.set_ylabel(None)
        fig.supylabel('Residence Time (a.u.)',**fontparams,x=0.05)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Free', 'Surface', 'Core'],
                            fontsize=18,fontweight='light')
        """
        if j == 1:
            ax.annotate(text=f'{Fore.RED}{ratio_free}{Style.RESET_ALL}:{Fore.RED}{ratio_surf}{Style.RESET_ALL}:{Fore.RED}{ratio_core}{Style.RESET_ALL}%',
                        xycoords="axes fraction",xy=(0.3,0.8),fontsize=15)"""

        ax.tick_params(axis='y',
                       labelsize=18)

        y = 0.95 if j == 1 else 1.08
        """ax.set_title(r'$U_{ns} = $' + f'{kts[i]}kT',
                     fontsize=20,
                     style='italic',
                     fontweight='light',
                     fontname='Arial',
                     y=y)"""
        
        if j==0:
            lower,upper = ax.get_ylim()
            ax.set_ylim([lower,upper*1.5])

#overall figure setting and saving
sns.despine(trim=True)

fig.savefig('../Figures/fig5C.pdf',transparent=True,bbox_inches='tight')
#fig.savefig('../../res_figs/raws/fig5.pdf',transparent=True,bbox_inches='tight')
