# %%
import os
import numpy as np
import polars as pl
from lets_plot import *
LetsPlot.setup_html()


UMS = ["10","20","40","60"]
KTS = ["2.80","3.00","3.50","4.00"]

# %%
def make_trailing_zeros(arr:np.ndarray)->np.ndarray:
    first_zeros = arr.argmin(axis=1) # first occurences of zeros for each tf (row)
    for i,index in enumerate(first_zeros):
        arr[i,index:] = 0

    return arr

# %%
def bounds(arr:np.ndarray,trailing_zero:bool=True)->np.ndarray:
    t0 = arr[:,0] # get the first column t=0
    subbed = np.subtract(arr[:,:].T,t0).T # subtract first column from all columns
    linear_distance = np.abs(subbed) # take the absolute value to get distance
    linear_distance[np.isnan(linear_distance)] = -1 # convert nan to -1 (if remote)
    linear_distance = linear_distance.astype(np.int16) # convert to i16 for tf_indeces are int
    status = linear_distance.copy() # steady (0), close(1) or remote(-1)
    status[(status>1)] = -1 # convert unbounds to -1
    bound = np.zeros_like(status) # from status to bound or not
    bound[(np.isin(status,[0,1]))] = 1 # rest will remain unbound (0)
    if trailing_zero:
        bound = make_trailing_zeros(bound) # once unbound the rest is unbound
    return bound

# %%
def remaining(arr:np.ndarray)->np.ndarray:
    arr = bounds(arr)
    counts = np.sum(arr,axis=0) # eaxtract counts of remaining bounds
    return counts

# %%
def bounds_dataframe(arr_L:np.ndarray,arr_R:np.ndarray,prefix:str="case")->pl.DataFrame:
    n_tf,n_timestep = arr_L.shape 
    df = pl.DataFrame()

    # bound status
    bound_all = make_trailing_zeros(bounds(arr_L,trailing_zero=False)+bounds(arr_R,trailing_zero=False))
    bound_all[bound_all==2] = 1

    count_L = remaining(arr_L)
    count_R = remaining(arr_R)
    count_all = np.sum(bound_all,axis=0)
    
    df = df.with_columns(
        pl.Series(count_L).alias(f"{prefix}_left"),
        pl.Series(count_R).alias(f"{prefix}_right"),
        pl.Series(count_all).alias(f"{prefix}_all"),
    )

    return df

# %%
def decay_plot(df:pl.DataFrame,until:int=20):

    df_melted = df.head(until).melt(id_vars="timestep",value_name='counts',variable_name="case")

    p = (
        ggplot(df_melted,aes(x='timestep',y='counts',color='case'))
        + geom_point(size=8)
        + geom_line(size=2.5)
        + scale_color_hue()
    )
    return p

# %%
def main():

    kt = "3.50"

    dfs = []
    for kt in KTS:
        for um in UMS:
            path = os.path.expanduser(f"~/5x10t/{kt}/{um}/positions.npz")
            positionz = np.load(path)
            L = positionz["L"]
            R = positionz["R"]
            df = bounds_dataframe(arr_L=L,arr_R=R,prefix=f"{kt}_{um}")
            dfs.append(df)
    
    nrows = df.height
    merged = pl.concat(dfs,how="horizontal").with_columns(
        pl.Series(np.arange(1,nrows+1)).alias('timestep'),
    )

    merged.write_csv("decays.csv")
    return

if __name__ == '__main__':
    main()
