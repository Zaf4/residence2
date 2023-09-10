import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from rich.progress import track
import matplotlib.colors as mcolors

# suppress the warning
pd.set_option('mode.chained_assignment', None)


def scatterit_multi(df: pd.DataFrame, fits: pd.DataFrame,
                    i:int, j:int, axes,
                    palette: str, **kwargs) -> None:
    """generates scatter plot with fits

    Parameters
    ----------
    df : pd.DataFrame
        Distribution data
    fits : pd.DataFrame
        Fit data
    i : int
        row of axes
    j : int
        column of axes
    axes : plt.Axes
        _description_
    palette : str
        color palette for the plots
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

    # resetting indexes
    df.index = np.arange(len(df))+1
    fits.index = np.arange(len(df))+1

    #font settings
    font = {'family': 'Sans Serif',
            'weight': 'light',
            'size': 20,
            }
    
    #weight calculation to prevent overcrowding
    ts = np.array(df.timestep)
    # weights = ((1/(ts*ts[::-1]))*10**5)**2
    tmax = df.timestep.max()
    weights = np.abs(ts-tmax/2)
    weights = 10**(weights/tmax)**2

    #sampled data
    sampled_df = df.sample(frac=0.25,random_state=42,weights=weights)
    # print(len(sampled_df),len(df))

    #scattter plot (Data)------------------------------------------------------
    sns.scatterplot(data=sampled_df,
                    x='timestep',
                    y='value',
                    palette=palette,
                    hue='case',
                    hue_order=cols,
                    alpha=1,
                    s=200,
                    edgecolor='white', 
                    linewidth=0.15,
                    ax=ax,
                    **kwargs)

    # Create a darker version of the viridis palette
    # Get the original viridis color palette
    viridis_palette = sns.color_palette(palette, as_cmap=True)

    # Create a darker version of the viridis palette
    darker = sns.color_palette([tuple([min(1, c+0.2) for c in color]) for color in viridis_palette.colors])
    darker_palette = sns.color_palette(darker,as_cmap=True)[40::50]
    

    #Line plot (Fits)----------------------------------------------------------
    sns.lineplot(data=fits,
                 x='timestep',
                 y='value',
                 palette=darker_palette,
                 hue='case',
                 ax=ax,
                 linewidth=3.5,
                 linestyle='solid',
                 alpha=1,
                 **kwargs)
    



    #graph settings------------------------------------------------------------
    
    ax.tick_params(axis='both',labelsize=21)
    ax.get_legend().remove()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.72, 3.7e3])
    ax.set_ylim([0.5, 1e6])
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    # ax.set_xlabel('Duration (a.u.)', fontdict=font)
    # ax.set_ylabel('Occurence', fontdict=font)

    return


def generate_fit_graph(datafile:os.PathLike = './data/durations_minimized.csv',
                       keywords:list[str] = ['10','20','40','60'],
                       figname:str="noname")->mpl.figure.Figure:

    """_summary_
    Parameters
    ----------
    datafile : os.PathLike
        path/to/datafile
    keywords : list[str]
        keyword to look for in column names
    figname : str
        figure name to save the file with

    Returns
    -------
    mpl.figure.Figure
        complete figure
    """


    durations = pd.read_csv(datafile, index_col=None)
    eqnames = ['ED','DED','TED','QED','PED','Powerlaw']
    #eqnames = ['ED','DED','PED','Powerlaw'] #for smaller figures
    
    
    nrow, ncol = len(eqnames), len(keywords)

    #seaborn settings
    sns.set(style='ticks',
            rc={
                #'font.weight': 'bold',
                'font.family': 'sans-serif',
                'axes.spines.top': 'False',
                'axes.spines.right': 'False',
                'ytick.minor.size': '0',
                'xtick.minor.size': '0',
                'ytick.major.size': '5',
                'xtick.major.size': '5',
                'legend.frameon': False
                }
            )
    # initing subfigures
    fig, axes = plt.subplots(nrow, ncol, 
                             sharex=True,sharey=True,
                             figsize=(ncol*4, nrow*3))


    for i, eqname in track(enumerate(eqnames),total=len(eqnames)):
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
                                palette='mako_r',
                                i=i, j=j)

                # legend = [f'{x[-2:]}µM' for x in cols]
                
                

            else:
                scatterit_multi(partial_data, 
                                partial_fits,
                                axes=axes,
                                palette='viridis_r',
                                i=i, j=j)
                
                # legend = [f'{x[:4]}kT' for x in cols]
                


 
    # fig.legend(legend,loc=(0.20,0.92),fontsize=15,markerscale=1.4,
            #    labelspacing=0.25,edgecolor='k')  
    fig.supxlabel('Duration (a.u.)', fontsize=24,fontweight='light')
    fig.supylabel('Occurence (n)', fontsize=24,fontweight='light') 
    plt.tight_layout()
    plt.savefig(f'../Figures/{figname}.pdf',
                transparent=True, bbox_inches='tight')
   
    return fig


if __name__ == '__main__':
    #changing working directory to current directory name
    os.chdir(os.path.dirname(__file__))

    ums = ['10','20','40','60']
    kts = ['1.00','2.80','3.00','3.50','4.00']
    
    fig = generate_fit_graph(keywords=ums,figname = 'fig2A')#fig2
    fig = generate_fit_graph(keywords=kts,figname = 'SI-fig1')#SI-fig1

    # presentation figure
    # kts_reduced = ['2.80','3.50','4.00']
    # generate_fit_graph(keywords=kts_reduced,figname='sunu_scatter')

    
    #plt.show()
    """
    datafile = './data/durations_minimized.csv'
    df = pd.read_csv(datafile, index_col=None)
    df['timestep'] = np.arange(len(df))+1
    #weight calculation to prevent overcrowding
    ts = np.array(df.timestep)
    # weights = ((1/(ts*ts[::-1]))*10**5)**2
    tmax = df.timestep.max()
    weights = ts**2-tmax*ts+(tmax/2)**2

    weights = weights**4/np.sum(weights)

    fig,axes = plt.subplots(1,3)
    print(np.min(weights))
    sns.lineplot(x=ts,y=weights,ax=axes[0])

    sns.lineplot(x=ts,y=weights**4,ax=axes[1])
    
    sns.lineplot(x=ts,y=np.log(weights),ax=axes[2])
    plt.show()
    """
