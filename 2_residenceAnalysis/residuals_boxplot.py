import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os


# docstring from _suplabels...
info = {'name': '_supylabel', 'x0': 0.1, 'y0': 0.5,
        'ha': 'left', 'va': 'center', 'rotation': 'vertical',
        'rotation_mode': 'anchor'}

def residuals2boxplot(residuals:pd.DataFrame,ax:plt.Axes,hue:str=None,
                      palette:str='flare')->None:
    """Create boxplots from residuals dataframe

    Parameters
    ----------
    residuals : pd.DataFrame
        residual values
    ax : plt.Axes
        current ax
    hue : str, optional
        hue, by default None
    palette : str, optional
        color palette for the current ax, by default 'flare'
    """
    
    #upper and lower boundaries for the aesthetics
    upper = 10**np.ceil(np.log10(residuals.value.max())) #upper boundary of boxplot
    lower = 10**np.floor(np.log10(residuals.value.min())) #lower boundary
    
    #kwargs for the boxplot
    props = {
    'boxprops':{'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
    
    }
    #boxplot for the residual comparison
    sns.boxplot(residuals,y='value',hue = hue,x='variable',
                palette=palette, saturation=1,linewidth=1.6,ax=ax,
                showfliers=False,**props)
    #labeling,modyifying and scaling the axes
    ax.legend(title=None,fontsize=18,loc='upper center')
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_yscale('log')
    ax.set_ylim([lower,upper])
    ax.tick_params(axis='x',labelsize=20,rotation=30)
    ax.tick_params(axis='y',labelsize=22)
    

    # plt.close()
    return

if __name__ == '__main__':

    #changing working directory to current directory name
    os.chdir(os.path.dirname(__file__))
    
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
    
    #plot settings
    sns.set(style='ticks',
    rc = {
            #'font.weight':'light',
            'font.family':'sans-serif',
            'axes.spines.top':'False',
            'axes.spines.right':'False',
            'ytick.minor.size':'0',
            'ytick.major.size':'10',
            'xtick.major.size':'10',
            'legend.frameon':False,
            'legend.fancybox':True,
            'legend.edgecolor':'black',
            # 'legend.linewidth':'1.6'
            
            
            }
    )
    #plotting section

    fig,axes = plt.subplots(nrows=3,ncols=1,figsize=(8,20))

    #graph by energy
    residuals2boxplot(res_melted,
                      hue='energy',
                      palette='viridis_r',
                      ax=axes[2])
    #graph by concentration
    residuals2boxplot(res_melted,
                      hue='concentration',
                      palette='mako_r',
                      ax=axes[1])
    
    #graph by concentration
    residuals2boxplot(res_melted,
                      palette='husl',
                      hue=None,
                      ax=axes[0])
    
    fig.supylabel('Normalized Residual Sum of Squares',fontsize=24,fontfamily='sans-serif',x=-0.02)
    
    ##saving the figure
    fig.savefig('../Figures/fig2B.pdf',
                transparent=True,
                bbox_inches='tight')