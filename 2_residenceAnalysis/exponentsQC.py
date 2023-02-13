import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def scatterit(fits:pd.DataFrame,palette:str)->plt.Axes:
    """
    

    Parameters
    ----------
    df : pd.DataFrame
        Main data.
    fits : pd.DataFrame
        Equation fit of the main data.
    color: list
        Colors range.

    Returns
    -------
    ax : plot
        seaborn scatter graph.

    """
    
    # df.index = np.arange(len(df))+1
    fits.index = np.arange(len(fits))+1
    ax = plt.figure()
    font = {'family': 'Arial',
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
                  'xtick.minor.size':'0',
                  'ytick.major.size':'10',
                  'xtick.major.size':'10',
                  'legend.frameon':False
                  
                  }
            )
     
    sns.set_palette(palette)
    # for i,col in enumerate(df):
    #     #scatterplot
    #     sns.scatterplot(data=df[col],s=50,
    #                     edgecolor='black',linewidth=0.25)

    sns.lineplot(data=fits,linewidth=5,
                 linestyle='dashed',alpha=1)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlim([0.72,1e2])
    plt.ylim([1,1e6])
    plt.xlabel('Duration (a.u.)',fontdict=font)
    plt.ylabel('Occurence',fontdict=font)
    plt.legend()
    return ax

if __name__ == '__main__':
    ex = pd.read_csv('./data/exponents.csv',index_col=None,
                     encoding= 'unicode_escape')
    
    xx = np.arange(24000)+1
    
    
    taus = [x for x in ex.columns if 'tau' in x]
    coeffs = [x for x in ex.columns if 'coeff' in x]
    
    taus = ex.iloc[90][taus]
    coeffs = ex.iloc[90][coeffs]
    
    sed = pd.DataFrame()
    for i,(tau,coeff) in enumerate(zip(taus,coeffs)):
        x1 = -np.log(1/coeff)*tau
        print(x1)
        yy = np.zeros([24000])
        for x in xx:
            yy[x-1] = coeff*np.exp(-x*(1/tau))
            
        
            
        sed[f'tau{i+1}'] = yy
     
    sed = sed.dropna(axis=1,how='all')
    arr = np.array(sed)
    sums = np.sum(arr,axis=1)
    sed['sums'] = sums
    
    
    ax = scatterit(sed,palette='bright')
    plt.savefig('taus.png',dpi=400,bbox_inches='tight')
    
    print('Plot shows shows me, fastest exponent is the one reaching to 1 first')
    print('So, A*exp(-x/tau)=1')
    print('...,x=-ln(1/A)*t')
    print('so I should sort by this value.. higher-->lower')
    
