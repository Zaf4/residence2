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

    data = pd.concat([data,data,data,data,data,data])

    data['equation'] = fits.equation

    data,fits = sample_df(data),sample_df(fits)

    ax = (
            ggplot(fits,aes('timestep','value',color='concentration'))+
                geom_point(data=data)+
                geom_line()+
                scale_x_log10(format='g',limits=[0.7,30000])+
                scale_y_log10(format='g',limits=[0.7,1_000_000])+
                scale_color_viridis()+
                theme_minimal()+
                facet_grid(x='energy',y='equation')
    )

    ggsave(ax,'sample.html')

    return


if __name__ == '__main__':
    #changing working directory to current directory name
    os.chdir(os.path.dirname(__file__))
    plot_dfs_facet()
