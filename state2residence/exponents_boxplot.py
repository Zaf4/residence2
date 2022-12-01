import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def boxplotit(residues:pd.DataFrame,graphname:str,palette='flare'):
    #residues of equations
    ax = plt.figure()
    
    font = {'family': 'Arial',
            'weight': 'light',
            'size': 14,
            }
    sns.set(style='ticks',
        rc = {'figure.figsize':(5.6,4.2),
              'font.weight':'light',
              'font.family':'Arial',
              'axes.spines.top':'False',
              'axes.spines.right':'False',
              'ytick.minor.size':'0',
              'ytick.major.size':'10',
              'xtick.major.size':'10'
              
              }
        )
    
    #kwargs for the boxplot
    props = {
    'boxprops':{'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
    
    }
    
    #boxplot for the residual comparison
    sns.boxplot(data=residues,palette=palette,
                saturation=1,linewidth=0.7,showfliers=False,
                **props)

    plt.ylabel('Σ(Residuals Normalized)²',fontdict=font)
    plt.yscale('log')
    # plt.ylim([1e-1,1e5])
    plt.savefig(f'./boxplots/{graphname}.png', dpi=400,
                # transparent=True,
                bbox_inches='tight')

    # residues.to_csv('residual.csv',index=False)

    
    return ax


ex = pd.read_csv('./data/exponents.csv',index_col=None)
ex = ex.dropna(axis=1,how='all')
exp_only = ['b','d','f','h','j']

boxplotit(ex[ex.concentration==40][exp_only],graphname='whole',palette='husl')