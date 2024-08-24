import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# suppress the warning
warnings.filterwarnings("ignore")
pd.set_option("mode.chained_assignment", None)


def sample_ends_favored(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """sample a dataframe but favor the ends

    Parameters
    ----------
    df : pd.DataFrame

    n : int, optional
        end size, by default 10

    Returns
    -------
    pd.DataFrame
        sampled dataframe
    """

    df = df.dropna(axis=0, how="any")

    # weight calculation to prevent overcrowding
    ts = np.array(df.timestep)
    # # weights = ((1/(ts*ts[::-1]))*10**5)**2
    tmax = df.timestep.max()
    weights = np.abs(ts - tmax / 2)
    weights = 10 ** (weights / tmax) ** 2
    sampled_df = pd.concat(
        [df.head(n), df.sample(frac=0.20, random_state=42, weights=weights), df.tail(n)]
    )
    return sampled_df





if __name__ == "__main__":
    # changing working directory to current directory name
    os.chdir(os.path.dirname(__file__))

    # values dataframe
    df = pd.read_csv("./data/duration_cont.csv", index_col=None)
    df = df_minimize(df, method="median")
    df[df == 0] = np.nan
    ts = np.arange(len(df)) + 1
    df["timestep"] = ts

    df.rename(
        columns={
            "uniform_250": "1-4kT Uniform",
            "uniform_350": "2-5kT Uniform",
            "gaus_250": "1-4kT Normal",
            "gaus_350": "2-5kT Normal",
        },
        inplace=True,
    )

    # reference line
    ref = df[["timestep", "3.50_60"]]

    for i, eqx in enumerate([tri_exp, powerlaw, powerlaw]):
        fits = pd.DataFrame()
        part = pd.DataFrame()
        col_names = []

        # creating fits and partial data df
        for col in df:
            if i == 2:
                fits[col], _, _ = value_fit(
                    np.array(df[col]), eq=eqx, sigma_w=True
                )  # fitting part
            else:
                fits[col], _, _ = value_fit(np.array(df[col]), eq=eqx)  # fitting part
            part[col] = df[col]
            col_names.append(col)
