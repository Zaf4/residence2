import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from rich.progress import track
import matplotlib.colors as mcolors
from lets_plot import *

LetsPlot.setup_html()

# suppress the warning
pd.set_option('mode.chained_assignment', None)


def melt_df(df:pd.DataFrame)->pd.DataFrame:
    #preprocess raw data and fits for graph------------------------------------
    
    df = pd.melt(df,
                 id_vars='timestep',
                 var_name='case',
                 value_name='value')

    return df


def sample_df(df:pd.DataFrame)->pd.DataFrame:
    #weight calculation to prevent overcrowding
    ts = np.array(df.timestep)
    # weights = ((1/(ts*ts[::-1]))*10**5)**2
    tmax = df.timestep.max()
    weights = ts**2-tmax*ts+(tmax/2)**2
    weights = weights**16/np.sum(weights)

    #sampled data
    sampled_df = df.sample(frac=0.08,random_state=42,weights=weights)
    return sampled_df

def plot_df(df:pd.DataFrame,fit:pd.DataFrame)->pd.DataFrame:
    ax=(
         ggplot(df,aes('timestep','value',color='case'))+
            geom_point()+
            geom_line(data=fit)+
            scale_x_log10(format='g')+
            scale_y_log10(format='g')+
            scale_color_viridis()+
            theme_classic()
    )
    return ax


def add_kt_um(df):
    ums = [f'{c[-2:]}µM' for c in df.case]
    kts = [f'{k[:4]}kT' for k in df.case]

    df['energy'] = kts
    df['concentration'] = ums

    return df



def unite_fits(eqnames:str = ['ED','DED','TED','QED','PED','Powerlaw'])->pd.DataFrame:
   
    full = pd.DataFrame()

    for i, eqname in track(enumerate(eqnames),total=len(eqnames)):

        fitfile = f'./data/{eqname}.csv'
        fits = pd.read_csv(fitfile, index_col=None)
        fits['timestep'] = np.arange(len(fits))+1

        #melting
        fits = pd.melt(fits,id_vars='timestep',var_name='case',value_name='value')

        # adding new equation,energy,concentration columns
        fits['equation'] = [eqname]*len(fits)
        fits = add_kt_um(fits)

        full = pd.concat([full,fits])

    
    full = full.drop(columns='case')

    return full

def pp_durations(datafile = './data/durations_minimized.csv'):
    """Preprocess the durations by meadding timestep then melting

    Returns
    -------
    _type_
        _description_
    """
    durations = pd.read_csv(datafile, index_col=None)
    durations['timestep'] = np.arange(len(durations))+1
    durations = melt_df(durations)
    durations = add_kt_um(durations)
    durations = durations.drop(columns='case')

    return durations

def plot_dfs_facet()->None:
    fits = unite_fits()
    data = pp_durations()

    data
    

    ax = (
            ggplot(fits,aes('timestep','value',color='concentration'))+
                geom_line()+
                scale_x_log10(format='g',limits=[0.7,30000])+
                scale_y_log10(format='g',limits=[0.7,1_000_000])+
                scale_color_viridis()+
                theme_classic()+
                facet_grid(x='energy',y='equation')
    )

    ggsave(ax,'sample.svg')


def generate_fit_graph(datafile:os.PathLike = './data/durations_minimized.csv',
                       keywords:list[str] = ['10','20','40','60'],
                       figname:str="noname"):

    """
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
    timestep = np.arange(len(durations))+1
    for i, eqname in track(enumerate(eqnames),total=len(eqnames)):
        fitfile = f'./data/{eqname}.csv'
        fits = pd.read_csv(fitfile, index_col=None)
        for j, keyword in enumerate(keywords):

            # columns wtih the keyword
            cols = [x for x in durations if keyword in x]
            data = durations[cols]
            fit = fits[cols]

            data['timestep'],fit['timestep'] = timestep,timestep

            data =melt_df(sample_df(data))
            fit = melt_df(sample_df(fit))

            if '.' in keyword:  # if for energy
                ax = plot_df(data, fit)
            else:
                ax =plot_df(data, fit)
                
                # legend = [f'{x[:4]}kT' for x in cols]
                
        ggsave(ax,figname)
        ax=""
    return


if __name__ == '__main__':
    #changing working directory to current directory name
    os.chdir(os.path.dirname(__file__))
    plot_dfs_facet()
    # ums = ['10','20','40','60']
    # kts = ['1.00','2.80','3.00','3.50','4.00']
    
    # fig = generate_fit_graph(keywords=ums,figname = 'fig2A-lp.svg')#fig2
    # fig = generate_fit_graph(keywords=kts,figname = 'SI-fig1-lp.svg')#SI-fig1

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
