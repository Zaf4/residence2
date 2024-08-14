# ..
import numpy as np
#import os
import polars as pl


KTS = ["2.80", "3.00", "3.50", "4.00"]
UMS = ["10", "20", "40", "60"]


def merge_legs(arr:np.ndarray)->np.ndarray:
    both = arr[::2]+arr[1::2]
    both[both>1] = 1
    return both

def count_bounds(arr:np.ndarray,)->np.ndarray:
    start = arr.sum(axis=0)[:1000].argmax()
    bound = np.where(arr[:,start]>0)[0]


    # ..
    bound = np.where(arr[:,start]>0)[0].tolist()
    reduced = arr[:,start:]

    # ..
    n_bound = [len(bound)] 
    for i in range(1, reduced.shape[1]):
        not_bound = np.where(reduced[:,i]==0)[0].tolist()
        for x in not_bound:
            if x in bound:
                bound.remove(x)
        n_bound.append(len(bound))


    final_arr = np.zeros(24_000) # 24 000 is the number of timesteps 
    n_elements = len(n_bound)
    final_arr[:n_elements] = n_bound
    final_arr = final_arr/final_arr.max()*100 # as percentage
    final_arr[final_arr==0] = np.nan

    return final_arr

def main():

    df = pl.DataFrame()

    for kt in KTS:
        for um in UMS:
            file_path = f"./5x10t/{kt}/{um}/states.npz"
            print(file_path)
            col_name = f"{kt}_{um}"
            states = np.load(file_path)
            _, sp = states["ns_state"], states["sp_state"]
            arr = merge_legs(sp)
            counts = count_bounds(arr)
            df = df.with_columns(pl.Series(counts).alias(col_name))

    df.write_parquet("./5x10t/FD.parquet")
    return

if __name__ == "__main__":
    main()