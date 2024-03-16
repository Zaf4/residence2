import pandas as pd
import numpy as np
from capsid2df import data2df
from lets_plot import *
LetsPlot.setup_html()
import os


def find_distance(arr:np.ndarray)->np.ndarray:
    size = len(arr)
    mtx = np.zeros([size,size])
    for i,a in enumerate(arr):
        mtx[:,i] = np.linalg.norm(arr-a,axis=1)
    
    return mtx

def mtx_melt(mtx:np.ndarray)->pd.DataFrame:
    df = pd.DataFrame(mtx)
    df['i'] = df.index
    dfm = df.melt(id_vars='i',var_name='j',value_name='Distance')
    
    return dfm

def heatmap(dfm,*,option="magma",direction=1):
    plot = (
            ggplot(dfm,aes(x='i',y='j',fill='Distance'))+
            geom_tile()+
            scale_fill_viridis(option=option,direction=direction)+
            ggsize(600,600)+
            #ggtitle('LETS PLOT')+
            #labs(x='Columns',y='Rows',caption='caption',title='title',subtitle='subtitle')+
            theme(title = element_text(hjust = 0.5))+
            scale_y_reverse()
            )
    
    return plot


def main():
    data_files = ['./data/data280.extra',
                 './data/data300.extra',
                 './data/data350.extra',
                 './data/data400.extra',]
    
    dfs = [data2df(file) for file in data_files]
    dna_only = [df[(df.type==1)|(df.type==2)] for df in dfs]
    xyz_arrs = [df[['x','y','z']].to_numpy() for df in dna_only]
    mtxs = [find_distance(arr) for arr in xyz_arrs]
    dfms = [mtx_melt(mtx) for mtx in mtxs]
    plots = [heatmap(dfm) for dfm in dfms]

    ggsave(gggrid(plots,ncol=2))

    return

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    if not os.path.exists('./figures'):
        os.mkdir('./figures')
    main()