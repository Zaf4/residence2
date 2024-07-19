from lets_plot import *
import numpy as np
import pandas as pd
import os
import polars as pl
from system_visualize import data2df, cluster_single_df


df_all = pl.DataFrame()
for case in [100,280,300,350,400]:
    coor = data2df(f'./data/data{case}.extra')
    df_cluster = cluster_single_df(coor)
    df = pl.from_pandas(df_cluster)
    df = df.with_columns(pl.col("clusterID").fill_null(0).cast(pl.Int32),
                pl.col(['type',"atomID"]).cast(pl.Int32))
    print(case)
    df = df.filter(pl.col('type') == 5, pl.col('clusterID')>0) 
    counts = df.group_by('clusterID').agg(pl.col('clusterID').count().alias('count')).sort('count').with_columns(pl.lit(f'{case/100}').alias('kT'))
    df_all = pl.concat([df_all, counts])
df_all

