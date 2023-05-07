import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def scatterit_multi(df: pd.DataFrame, fits: pd.DataFrame,
                    i:int, j:int, axes,
                    palette: str, **kwargs) -> plt.Axes:
    """


    Parameters
    ----------
    df : pd.DataFrame
        Main data.
    fits : pd.DataFrame
        Equation fit of the main data.

    Returns
    -------
    ax : plot
        seaborn scatter graph.

    """
    
    #preprocess raw data and fits for graph------------------------------------
    cols = [col for col in df]
    df['timestep'] = np.arange(len(df))+1
    fits['timestep'] = np.arange(len(df))+1
    
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
    
    ax = axes[i,j]
    
    if i==0 and j == 0:
        legend = True
    else:
        legend = False

    # resetting indexes
    df.index = np.arange(len(df))+1
    fits.index = np.arange(len(df))+1

    # seting seaborn parameters
    font = {'family': 'Sans Serif',
            'weight': 'light',
            'size': 14,
            }

    sns.set(style='ticks',
            rc={
                'font.weight': 'light',
                'font.family': 'sans-serif',
                'axes.spines.top': 'False',
                'axes.spines.right': 'False',
                'ytick.minor.size': '0',
                'xtick.minor.size': '0',
                'ytick.major.size': '10',
                'xtick.major.size': '10',
                'legend.frameon': False

                }
            )
    
    #weight calculation to prevent overcrowding
    ts = np.array(df.timestep)
    weights = ((1/(ts*ts[::-1]))*10**5)**2
    print(weights[::12000])
    
    #scattter plot (Data)------------------------------------------------------
    sns.scatterplot(data=df.sample(frac=0.1,random_state=42,weights=weights),
                    x='timestep',
                    y='value',
                    palette=palette,
                    hue='case',
                    alpha=1,
                    s=40,
                    edgecolor=None, 
                    linewidth=0.01,
                    ax=ax,
                    legend=legend,
                    **kwargs)
    #Line plot (Fits)----------------------------------------------------------
    """sns.lineplot(data=fits,
                 x='timestep',
                 y='value',
                 palette=palette,
                 hue='case',
                 ax=ax,
                 linewidth=3,
                 linestyle='dashed',
                 alpha=1, 
                 **kwargs)"""

    #graph settings------------------------------------------------------------    
    ax.tick_params(axis='both',labelsize=12)
    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.72, 3.7e3])
    ax.set_ylim([0.5, 1e6])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    # ax.set_xlabel('Duration (a.u.)', fontdict=font)
    # ax.set_ylabel('Occurence', fontdict=font)

    return


def generate_fit_graph(datafile:str = './data/durations_minimized.csv',
                       keywords:list[str] = ['10','20','40','60']) -> None:
    """


    Parameters
    ----------

    fitfile : str
        path/to/fitfile.

    Returns
    -------
    None

    """

    
    durations = pd.read_csv(datafile, index_col=None)

    eqnames = ['ED','DED','TED','QED','PED','Powerlaw']
    

    nrow, ncol = len(eqnames), len(keywords)

    # initing subfigures
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol*4, nrow*3))

    for i, eqname in enumerate(eqnames):
        for j, keyword in enumerate(keywords):

            fitfile = f'./data/{eqname}.csv'

            fits = pd.read_csv(fitfile, index_col=None)
            fname = fitfile.split('/')[-1]
            fname = fname[:-4]

            # columns wtih the keyword
            cols = [x for x in durations if keyword in x]
            partial_data = durations[cols]
            partial_fits = fits[cols]

            if '.' in keyword:  # if for energy
                scatterit_multi(partial_data,
                                partial_fits,
                                axes=axes,
                                palette='mako_r')
                # plt.text(x=1,y=1,color='grey',
                #          s=f'Fit equation:{fname}\nEnergy:{keyword}kT')
                legend = [f'{x[-2:]}µM' for x in cols]
                # keyword = ''.join(keyword.split('.'))

            else:
                scatterit_multi(partial_data, 
                                partial_fits,
                                axes=axes,
                                palette='viridis_r',
                                i=i, j=j)
                # plt.text(x=1,y=1,color='grey',
                #          s=f'Fit equation:{fname}\nConcentration:{keyword}µM')
                legend = [f'{x[:4]}kT' for x in cols]

            # plt.legend(legend)
            # creating folder(if not already there), saving the graph
            if not os.path.exists('./scatters'):
                os.mkdir('scatters')
                
    fig.legend(legend,loc=(0.15,0.92),fontsize=15,markerscale=1.4,
               labelspacing=0.25)   
    plt.tight_layout()
    plt.savefig('./scatters/multi_scatter_kT.pdf',
                transparent=True, bbox_inches='tight')
   
    # plt.close()


if __name__ == '__main__':

    
    ums = ['10','20','40','60']
    kts = ['1.00','2.80','3.00','3.50','4.00']
    
    
    generate_fit_graph(keywords=ums)

    



    
    
    
    