import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def residuals2boxplot(residuals:pd.DataFrame,hue_by=None,
                      graphname:str='residuals',palette:str='flare'):
    #residues of equations
    ax = plt.figure()
    
    #upper and lower boundaries for the aesthetics
    upper = 10**np.ceil(np.log10(residuals.value.max())) #upper boundary of boxplot
    lower = 10**np.floor(np.log10(residuals.value.min())) #lower boundary
    
    
    #font and style setttings
    font = {'family': 'sans-serif',
            'weight': 'light',
            'size': 14,
            }
    sns.set(style='ticks',
        rc = {'figure.figsize':(5.6,4.2),
              'font.weight':'light',
              'font.family':'sans-serif',
              'axes.spines.top':'False',
              'axes.spines.right':'False',
              'ytick.minor.size':'0',
              'ytick.major.size':'10',
              'xtick.major.size':'10',
              'legend.frameon':False
              
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
    sns.boxplot(residuals,y='value',hue = hue_by,x='variable',
                        palette=palette, saturation=1,linewidth=0.7,
                        showfliers=False,**props)
    #labeling,modyifying and scaling then saving
    plt.legend(title=None)
    plt.xlabel(None)
    plt.ylabel('(ΣResiduals²)/N(Values)',fontdict=font)
    plt.yscale('log')
    plt.ylim([lower,upper])
    
    #creating folder(if not already there), saving the graph
    if not os.path.exists('./boxplots'):
        os.mkdir('boxplots')
    plt.savefig(f'./boxplots/{graphname}_by_{hue_by}.png', dpi=400,
                # transparent=True,
                bbox_inches='tight')
    # plt.close()
    return

if __name__ == '__main__':
   
    residuals = pd.read_csv('./data/residuals.csv',index_col=0)
    
    kts = [f'{x[:4]}kT' for x in residuals.index]
    ums = [f'{x[-2:]}µM' for x in residuals.index]
    
    #newcolumns
    residuals['energy'] = kts
    residuals['concentration'] = ums
    
    #melting the dataframe
    res_melted = pd.melt(residuals,
                         value_vars=['ED','DED','TED','QED','PED','Powerlaw'],
                         id_vars=['energy','concentration'])
    
    #graph by energy
    residuals2boxplot(res_melted,
                      graphname='residuals',
                      hue_by='energy',
                      palette='viridis_r')
    #graph by concentration
    residuals2boxplot(res_melted,
                      graphname='residuals',
                      hue_by='concentration',
                      palette='mako_r')
    
    #graph by concentration
    residuals2boxplot(res_melted,
                      graphname='residuals_averaged',
                      palette='husl',
                      hue_by=None)
    
