from lets_plot import *
import lets_plot
import numpy as np
import pandas as pd
import os
import polars as pl
from system_visualize import data2df, cluster_single_df


KTS = ["1.00", "2.80", "3.00", "3.50", "4.00"]
UMS = ["10", "20", "40", "60"]

def extract_counts(kt: str, um:int)->pl.DataFrame:
    """
    extract number of clusters for each cluster
    with type 5 and clusterID>0
    with given kt (energy) and um (concentration)

    Parameters
    ----------
    df : pl.DataFrame
        dataframe to extract from.
    kt : str
        energy.
    um : int
    """
    print(f'{kt=}, {um=}')
    coor = data2df(f'./data/data{kt}_{um}.extra') # read as the dataframe
    df_cluster = cluster_single_df(coor) # find clusters in the dataframe
    df = pl.from_pandas(df_cluster) # convert to polars dataframe
    df = df.with_columns(pl.col("clusterID").fill_null(0).cast(pl.Int32),
                pl.col(['type',"atomID"]).cast(pl.Int32))
    df = df.filter(pl.col('type') == 5, pl.col('clusterID')>0) 
    counts = df.group_by('clusterID').agg(pl.col('clusterID').count().alias('count')
                                          ).sort('count').with_columns(
                                              pl.lit(f'{kt}').alias('kT'),
                                              pl.lit(um).alias('um')
                                              )
    
    return counts

def collect_dataframes(KTS: list, UMS: list)->pl.DataFrame:
    """
    Collect dataframes for each kT and um
    """
    df_all = pl.DataFrame()
    for kt in KTS:
        for um in UMS:
            counts = extract_counts(kt=kt,um=um)
            df_all = pl.concat([df_all, counts])
    return df_all

def cluster_size_vs_affinity(df: pl.DataFrame):
    """_summary_

    Parameters
    ----------
    df : pl.DataFrame
        dataframe to visualize.
        with cluster sizes, affinities and concentrations.
    """
    

    graph = (
        ggplot(df, aes(x='kT', y='count', fill='um')) +
        geom_boxplot()+
        scale_fill_viridis()+
        labs(title='', x='Affinity (kT)', y='Cluster size')+
        theme_classic()
    )

    return graph

def cluster_size_vs_concentration(df: pl.DataFrame):
    """_summary_

    Parameters
    ----------
    df : pl.DataFrame
        dataframe to visualize.
        with cluster sizes, affinities and concentrations.
    """
    

    graph = (
        ggplot(df, aes(x='um', y='count', fill='kT')) +
        geom_boxplot()+
        scale_fill_viridis()+
        labs(title='', x='Concentration (µM)', y='Cluster size')+
        theme_classic()
    )

    return graph



def main():
    df = collect_dataframes(KTS, UMS)
    affinity_graph = cluster_size_vs_affinity(df)
    concentration_graph = cluster_size_vs_concentration(df)
    ax = gggrid(affinity_graph, concentration_graph)
    ggsave(ax, path='..Figures/fig3DE.pdf', width=8, height=6)

    return

if __name__ == '__main__':
    main()