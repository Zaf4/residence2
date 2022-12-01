import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def residuals2boxplot(residuals:pd.DataFrame,unit:tuple,cpalette:str='flare'):
    #residues of equations
    ax = plt.figure()
    
    #upper and lower boundaries for the aesthetics
    upper = 10**np.ceil(np.log10(residuals.max().max())) #upper boundary of boxplot
    lower = 10**np.floor(np.log10(residuals.min().min())) #lower boundary
    
    #name for the plot
    if  unit[0]<10:#enetgies are all lower than 10
        name = f'{unit[0]}kT'
    else:#concentrations are at least ten
        name = f'{unit[0]}uM'
    
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
    sns.boxplot(data=residuals,palette=cpalette,
                saturation=1,linewidth=0.7,showfliers=False,
                **props)
    #labeling and scaling
    plt.ylabel('Σ(Residuals²)/N(Values)',fontdict=font)
    plt.yscale('log')
    plt.ylim([lower,upper])
    plt.savefig(f'./boxplots/{name}_residues.png', dpi=400,
                # transparent=True,
                bbox_inches='tight')
    plt.text(x=-0.2,y=lower*2,s=f'{unit[0]}{unit[1]}',color='grey',fontdict=font)

    # residues.to_csv('residual.csv',index=False)

    
    return ax

if __name__ == '__main__':
   
    residuals = pd.read_csv('./data/residuals.csv',index_col=0)
    
    kts = [float(x[:4]) for x in residuals.index]
    ums = [int(x[-2:]) for x in residuals.index]
    
    residuals['kT'] = kts
    residuals['µM'] = ums
    
    residuals = residuals.set_index(['kT','µM'])
    #grouping by energy and concentrations
    ktgroup = residuals.groupby('kT')
    umgroup = residuals.groupby('µM')
    
    for umg in umgroup:
        residuals2boxplot(umg[1],cpalette='magma_r',unit=(umg[0],'µM'))

    
    for ktg in ktgroup:
        residuals2boxplot(ktg[1],cpalette='magma_r',unit=(ktg[0],'kT'))
