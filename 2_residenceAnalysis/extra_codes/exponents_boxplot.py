import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def boxplotit(exponents:pd.DataFrame,hue_by=None,
              graphname:str='exponents',
              xval:str='tau',yval:str='value',
              yscale:str='log',palette='flare',
              ):
    

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

    ax = plt.figure()
    #normalization for log scale
    if yscale=='log':
        exponents.value +=0.1
        
        
    
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 18,
            }

    
    #kwargs for the boxplot
    props = {
    'boxprops':{'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
    
    }
    
    sns.boxplot(exponents,y='value',hue = hue_by,x=xval,
                        palette=palette, saturation=1,linewidth=0.7,
                        showfliers=False,**props)
    
    plt.legend(title=None,markerscale=0.6)
    plt.xlabel(None)
    plt.ylabel('τ:Mean Lifetime (a.u)',fontdict=font)
    #seting y scale and its boundaries
    plt.yscale(yscale)
    lower_auto,upper_auto = plt.gca().get_ylim()
    if yscale == 'linear':
        lower=0
        exp = 10**int(np.log10(upper_auto))
        upper = (int(upper_auto/exp)+1)*exp
    
    elif yscale == 'log':
        lower = 0.1
        exp = int(np.log10(upper_auto))
        upper = 10**(exp+1)
        
    plt.ylim([lower,upper])
    #creating folder(if not already there), saving the graph
    if not os.path.exists('./boxplots'):
        os.mkdir('boxplots')
    plt.savefig(f'./boxplots/{graphname}_{yval}vs{xval}_{hue_by}.png',
                # transparent=True,
                bbox_inches='tight')

    return ax

def tau_boxplots(ex_melted:pd.DataFrame,graphname:str='exponents'):
    #graphs with X = tau
    ##graph by energy 
    boxplotit(ex_melted,
              hue_by='energy',
              palette='viridis_r',
              yscale='log',
              graphname=graphname)
    ##graph by concentration
    boxplotit(ex_melted,
              hue_by='concentration',
              palette='mako_r',
              yscale='linear',
              graphname=graphname)
    
    
    #graphs with X = concentration or X=energy
    ##graph by energy
    boxplotit(ex_melted,
              xval='concentration', 
              hue_by='energy',
              palette='viridis_r',
              yscale='linear',
              graphname=graphname)
    ##graph by concentration
    boxplotit(ex_melted,
              xval='energy',
              hue_by='concentration',
              palette='mako_r',
              yscale='linear',
              graphname=graphname)

    #averages
    ##graph by energy
    boxplotit(ex_melted,
              xval='concentration', 
              palette='mako_r',
              yscale='linear',
              graphname=graphname)
    
    ##graph by concentration
    boxplotit(ex_melted,
              xval='energy',
              palette='viridis_r',
              yscale='linear',
              graphname=graphname)

if __name__ == '__main__':
    
    ex = pd.read_csv('./data/exponents.csv',index_col=None,
                     encoding= 'unicode_escape')
    
    ex_melted = pd.melt(ex,
                        var_name='tau',
                        value_vars=['tau1','tau2','tau3','tau4','tau5'],
                        id_vars=['equation','energy','concentration'])
    
    tau_boxplots(ex_melted,graphname='average')
    eqnames = ['ED','DED','TED','QED','PED','Powerlaw']
    
    """
    for eqname in eqnames:
        ex_partial = ex[ex.equation==eqname]
        ex_melted = pd.melt(ex_partial,
                            var_name='tau',
                            value_vars=['tau1','tau2','tau3','tau4','tau5'],
                            id_vars=['equation','energy','concentration'])
        
        tau_boxplots(ex_melted,graphname=eqname)
        
        """
    

    

    
