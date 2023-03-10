import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def scatterit(df:pd.DataFrame,fits:pd.DataFrame,
              palette:str,**kwargs)->plt.Axes:
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
    
    df.index = np.arange(len(df))+1
    fits.index = np.arange(len(df))+1
    
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
    for i,col in enumerate(df):
        #scatterplot
        sns.scatterplot(data=df[col],s=50,
                        edgecolor='black',linewidth=0.25,palette=palette,
                        **kwargs)
    for i,col in enumerate(fits):
        #lineplot
        sns.lineplot(data=fits[col],palette=palette,
                     linestyle='dashed',alpha=1,**kwargs)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlim([0.72,3.7e3])
    plt.ylim([0.5,1e6])
    plt.xlabel('Duration (a.u.)',fontdict=font)
    plt.ylabel('Occurence',fontdict=font)

    return 

def generate_fit_graph(datafile:str,fitfile:str,keyword:str,**kwargs)->None:
    """
    

    Parameters
    ----------
    datafile : str
        path/to/datafile.
    fitfile : str
        path/to/fitfile.

    Returns
    -------
    None

    """
    
    durations = pd.read_csv(datafile,index_col=None)
    fits = pd.read_csv(fitfile,index_col=None)
    fname = fitfile.split('/')[-1]
    fname = fname[:-4]
    
    cols = [x for x in durations if keyword in x] #columns wtih the keyword
    partial_data = durations[cols]
    partial_fits = fits[cols]
    
    
    if '.' in keyword: #if for energy
        ax = scatterit(partial_data,partial_fits,'mako_r',**kwargs)

        legend = [f'{x[-2:]}µM' for x in cols]
        keyword = ''.join(keyword.split('.'))

    else:
        ax = scatterit(partial_data,partial_fits,'viridis_r',**kwargs)

        legend = [f'{x[:4]}kT' for x in cols]
    
    plt.legend(legend)
    #creating folder(if not already there), saving the graph
    if not os.path.exists('./scatters'):
        os.mkdir('scatters')
    # plt.savefig(f'./scatters/{fname}_{keyword}.png',dpi=400,
    #           transparent=False,bbox_inches='tight')
    #plt.close()
    return
                    


if __name__ == '__main__':
    data = './data/durations_minimized.csv'
    #names and functions are listed for the for loop

    """
    eqnames = ['ED','DED','TED','QED','PED','Powerlaw']
    keywords = ['10','20','40','60','1.00','2.80','3.00','3.50','4.00']

    """


    eqnames = ['ED','DED','Powerlaw']
    keywords = ['10','20','40','60']

    nrow,ncol = len(eqnames),len(keywords)

    fig,axes = plt.subplots(nrow,ncol,
                            sharex=True,sharey=True,
                            figsize=(nrow*6,ncol*4))
    
    for i,eqname in enumerate(eqnames):
        for j,keyword in enumerate(keywords):
            generate_fit_graph(datafile='./data/durations_minimized.csv',
                               fitfile=f'./data/{eqname}.csv',
                               keyword=keyword,
                               ax = axes[i,j])

    sns.set_palette('viridis')    
    plt.setp(axes,
             xlim=[0.72,3.7e3],
             ylim=[0.5,1e6],
             xscale='log',
             yscale='log')
    
    plt.show()

    """
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim()
    plt.ylim([0.5,1e6])
    plt.xlabel('Duration (a.u.)',fontdict=font)
    plt.ylabel('Occurence',fontdict=font)"""