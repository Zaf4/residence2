import polars as pl
import numpy as np


def split(path="data/duration_cont.csv"):
    df = pl.read_csv(path)
    ref = df.select(pl.col("3.50_60"))
    # ref.write_csv("data/reference.csv")

    data = df.select(pl.all().exclude("3.50_60"))
    # df.write_csv("data/duration_dist_affinity.csv")
    return data, ref

def arr_minimize(arr: np.ndarray, method: str = "median") -> np.ndarray:
    """
    Minimizes 1d array by removing repeats, according to the given method.

    Parameters
    ----------
    arr : np.ndarray
        1d array to be minimized.
    method : str, optional
        'median' or 'average'. The default is 'median'.

    Returns
    -------
    arr1 : np.ndarray
        minimized array.

    """

    search = np.unique(arr)  # arr of unique elements
    search = search[search > 0]  # remove nans

    arr1 = arr.copy().astype(float)

    for s in search:
        (positions,) = np.where(arr == s)
        if method == "median":
            mid = int(np.median(positions))

        elif method == "average":
            mid = int(np.average(positions))
        elif method == "max":
            mid = int(np.max(positions))
        elif method == "min":
            mid = int(np.min(positions))

        arr1[arr1==0] = np.nan
        arr1[positions] = np.nan
        arr1[mid] = s  # mid value is kept

    return arr1

def df_minimize(df: pl.DataFrame, method: str = "median") -> pl.DataFrame:

    minimized_df = pl.DataFrame()
    for column in df.columns:
        arr = df.select(column).to_numpy().flatten()
        arr = arr_minimize(arr, method=method)
        minimized_df = minimized_df.with_columns(
            pl.Series(arr).alias(column)
        )
    
    return minimized_df


def add_timestep(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.int_range(1, df.height + 1).alias("timestep"))


def log_normalize(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.all().add(1).log10()) # plus 1 log10 


def melt(df: pl.DataFrame) -> pl.DataFrame:
    df = df.unpivot(
        index="timestep", value_name="remaining", variable_name="case"
    ).with_columns(
        pl.col("case").str.split("_").list.get(0).alias("type"),
        pl.col("case").str.split("_").list.get(1).alias("kt"),
    )

    return df

def remove_zeros(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("remaining") > 0)


def main():
    data, ref = split()

    data = df_minimize(data)
    data = add_timestep(data)    
    # data = log_normalize(data)

    # data = melt(data)
    # data = remove_zeros(data)

    data.write_csv("data/dist_minimized.csv")
    # ref.write_csv("data/ref.csv")

    return


if __name__ == "__main__":
    main()
