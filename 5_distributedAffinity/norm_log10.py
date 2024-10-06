import os
import warnings

import polars as pl
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erfc
# from math import erfc
from lets_plot import (
    LetsPlot,
    element_blank,
    theme,
    geom_point,
    geom_line,
    scale_x_log10,
    scale_y_log10,
    labs,
    scale_fill_brewer,
    scale_color_brewer,
    gggrid,
    ggplot,
    ggsize,
    lims,
    theme_classic, aes,
    scale_color_viridis,
    scale_fill_viridis,
    element_text,
    element_geom,
    scale_color_manual,
    scale_fill_manual,
    scale_fill_hue,
    scale_color_hue,
    geom_histogram,
    theme_void,
    geom_density,
)
LetsPlot.setup_html()


def double_exp(x, a, b, c, d):
    return a * np.exp(-x * b) + (d) * np.exp(-x * c)


def erfc_exp(t, A, tau, beta):
    return A * np.exp(-(t / tau) ** beta)


def erfc_exp(t, a, koff, ks):
    return (a 
            * np.exp(( np.square(koff) * t) / ks) 
            * erfc(np.sqrt((np.square(koff) * t) / ks))
    )

def powerlaw(x, a, b):
    return a * x ** (-b)


def exp_decay(x, a, b):
    return a * np.exp(-x * b)


def tri_exp(x, a, b, c, d, e, f):
    return a * np.exp(-x * b) + c * np.exp(-x * d) + e * np.exp(-x * f)


def quad_exp(x, a, b, c, d, e, f, g, h):
    return (
        a * np.exp(-x * b)
        + c * np.exp(-x * d)
        + e * np.exp(-x * f)
        + g * np.exp(-x * h)
    )


def penta_exp(x, a, b, c, d, e, f, g, h, i, j):
    return (
        a * np.exp(-x * b)
        + c * np.exp(-x * d)
        + e * np.exp(-x * f)
        + g * np.exp(-x * h)
        + i * np.exp(-x * j)
    )

def deleteNaN(y: np.ndarray,t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    delete NaN parts of the input array and time array opened for it,
    and returns time array and values array.

    """

    t = t[~np.isnan(y)]
    y = y[~np.isnan(y)]

    return t, y

def value_fit(
    val: np.ndarray, t: np.ndarray, tmax: int, 
    eq: callable,
    delete_nan: bool = True
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """

    Parameters
    ----------
    val : np.ndarray
        Values 1d array to fit.
    eq : callable
        Equation to create a fit.

    Returns
    -------
    y_fit : np.ndarray
        1d Fitted values array.
    ss_res_norm : np.ndarray
        Sum of squares of residuals normalized.
    popt : tuple

    """
    t_range = np.arange(tmax) + 1

    if delete_nan:
        t, val = deleteNaN(val,t)
        t, val = np.log10(t), np.log10(val)
    if eq.__name__ == "erfc_exp":
        p0=[1,1,1]
        popt, _ = curve_fit(eq, t, val, p0=p0, maxfev=200_000_000_0)
    else:
        popt, _ = curve_fit(eq, t, val, maxfev=200_000_000_0)
    print(f"{eq.__name__}: {popt}")
    

    y_fit = eq(t_range, *popt)  # full time length
    # y_fit[y_fit < 1] = np.nan  # too small values to be removed
    # y_fit[y_fit > np.max(val) * 2] = np.nan  # too big values removed

    return y_fit




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

    arr1 = arr.copy()

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

        arr1[positions] = np.nan
        arr1[mid] = s  # mid value is kept

    return arr1


def df_minimize(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be minimized.

    Returns
    -------
    df : pd.DataFrame
        Minimized DataFrame.

    """
    for i in range(len(df.columns)):
        df.iloc[:, i] = arr_minimize(
            df.iloc[:, i], **kwargs
        )  # values minimized and returned

    return df




def sample_ends_favored_1D(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
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


    mid_df = df.iloc[n:-n]
    sampled_df = pd.concat(
        [df.head(n), mid_df.sample(frac=0.20, random_state=42), df.tail(n)],
        ignore_index=True
    )
    return sampled_df


def sample_EF_1D(df: pl.DataFrame, n: int = 10) -> pd.DataFrame:
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

    

    mid_df = df.slice(n, len(df) - 2 * n)
    sampled_df = pl.concat(
        [df.head(n), mid_df.sample(fraction=0.20, seed=42), df.tail(n)],
        how="vertical",
    ).unpivot(index="timestep", 
              value_name="remaining", 
              variable_name="case")
    
    return sampled_df


def sample_EF_2D(df: pl.DataFrame, n: int = 10) -> pd.DataFrame:

    all_parts = [
        sample_EF_1D(df.select([col, "timestep"]), n=n).with_columns(
            pl.lit(col).alias("case")
        )
        for col in df.columns if col != "timestep"
    ]
    collected = pl.concat(all_parts, how="vertical")

    return collected.drop_nulls()

# values dataframe
def values_df() -> pl.DataFrame:
    df = pd.read_csv("./data/duration_cont.csv", index_col=None)
    df = df_minimize(df, method="median")
    df[df == 0] = np.nan
    df = df.dropna(axis=0, how="all")
    df["timestep"] = df.index + 1
    df = pl.from_pandas(df)
    return df


def add_type_um(df: pl.DataFrame) -> pl.DataFrame:

    return df.with_columns(
        pl.col("case").str.split("_").list.get(0).alias("type"),
        pl.col("case").str.split("_").list.get(1).alias("um"),
    )

def sample_df(df: pl.DataFrame, n: int = 40) -> pl.DataFrame:
    df = sample_EF_2D(df, n=n)
    df = df.with_columns(
        pl.col("case").str.split("_").list.get(0).alias("distribution"),
        pl.col("case").str.split("_").list.get(1).alias("around"),
    )
    return df


# df350 = df.filter(pl.col("case") == "3.50_60")
# df350 = df350.with_columns(pl.col("timestep").log1p(),pl.col("remaining").log1p())


df.filter(pl.col("case") == "uniform_350").sort("timestep").tail()
df.filter(pl.col("case") == "gaus_350").sort("timestep").tail()


cases = df.select("case").unique().to_series().to_list()
cases


df.select("timestep").to_series().max()


# make fit
tmax = df.select("timestep").to_series().max()
fits_arr = list()
fits = pl.DataFrame().with_columns(
    pl.int_range(1, tmax + 1).alias("timestep"),
)
cases = df.select("case").unique().to_series().to_list()
for c in cases:
    if not c == "3.50_60":
        case_df = df.filter(pl.col("case") == c)
        timestep = case_df.select("timestep").to_numpy()
        data = case_df.select("remaining").to_numpy()
    
        fits = fits.with_columns(
            pl.Series(value_fit(data, timestep, tmax=tmax, eq=tri_exp)).alias(f"{c}_tri.exp"),
            pl.Series(value_fit(data, timestep, tmax=tmax, eq=quad_exp)).alias(f"{c}_quad.exp"),
            pl.Series(value_fit(data, timestep, tmax=tmax, eq=powerlaw)).alias(f"{c}_powerlaw"),
        )


def make_fits(df: pl.DataFrame) -> pl.DataFrame:
    tmax = df.select("timestep").to_series().max()
    fits = pl.DataFrame().with_columns(
        pl.int_range(1, tmax + 1).alias("timestep"),
    )
    cases = df.select("case").unique().to_series().to_list()
    for c in cases:
        if not c == "3.50_60":
            case_df = df.filter(pl.col("case") == c)
            timestep = case_df.select("timestep").log1p().to_numpy()
            data = case_df.select("remaining").log1p().to_numpy()
        
            fits = fits.with_columns(
                pl.Series(value_fit(data, timestep, tmax=tmax, eq=tri_exp)).alias(f"{c}_tri.exp"),
                pl.Series(value_fit(data, timestep, tmax=tmax, eq=erfc_exp)).alias(f"{c}_erfc.exp"),
            )
    return fits


fitsm = fits.unpivot(index="timestep",
                     value_name="remaining",
                     variable_name="fit_case")


fitsm = fitsm.with_columns(
    pl.col("fit_case").str.split("_").list.get(0).alias("distribution"),
    pl.col("fit_case").str.split("_").list.get(1).alias("around"),
    pl.col("fit_case").str.split("_").list.get(2).alias("equation")
)


gauss = fitsm.filter(pl.col("distribution")=="gaus",pl.col("remaining").is_not_nan())
uni = fitsm.filter(pl.col("distribution")=="uniform",pl.col("remaining").is_not_nan())


gauss.write_csv('gaus.csv')
gauss.head(12)


df = df.with_columns(pl.col("timestep").log1p(),pl.col("remaining").log1p())
df.sample(12)


df_gauss =df.filter(pl.col("distribution")=="gaus")


eqs = ["tri.exp","erfc.exp"]


def plot_it(df:pl.DataFrame,fits:pl.DataFrame,color_by):

    return (ggplot()
            + geom_point(data=df, mapping=aes(x="timestep", y="remaining",color=color_by),fill="#dfdfdf",shape=21, stroke=1, size=5)
            + geom_line(data=fits, mapping=aes(x="timestep", y="remaining",color=color_by),size=3,alpha=1)
            + geom_line(data=df350,mapping=aes(x="timestep", y="remaining"),size=2,alpha=0.7, color="black")
            + scale_color_brewer(labels=["1-4kT", "2-5kT"])
            + scale_fill_brewer(labels=["1-4kT", "2-5kT"])
            + scale_x_log10(format="~e")
            + scale_y_log10(format=".0~e")
            + theme_classic()
            + ggsize(800,400)
            + lims(x=(0.72, 3.7e3),y=(0.5, 1.1e7))
            + theme(legend_title=element_blank(), 
                    exponent_format="pow",
                    axis_text=element_text(size=20,color="#1f1f1f"),
                    axis_title=element_text(size=21,color="#1f1f1f"),
                    legend_text=element_text(size=20),
                    legend_position=[0.8, 0.8]
                    )
            + labs(x="Duration (a.u.)", y="Occurence (n)")
            
            )




grid_gauss =[
        plot_it(df=df.filter(pl.col("distribution")=="gaus").sort("around"),
                fits=gauss.filter(pl.col("equation")==eq).sort("around"),
                color_by="around") 
                for eq in eqs 
                ]


grid_uni= [
        plot_it(df=df.filter(pl.col("distribution")=="uniform").sort("around"),
                fits=uni.filter(pl.col("equation")==eq).sort("around"),
                color_by="around")+ scale_fill_hue(labels=["1-4kT", "2-5kT"])+scale_color_hue(labels=["1-4kT", "2-5kT"])
                for eq in eqs 
                ]


all = gggrid(grid_gauss+grid_uni,ncol=2)+ggsize(900,700)
all


all.to_png('./fig6__ERFC.png')








