import polars as pl
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erfc
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

# import seaborn as sns
# import matplotlib.pyplot as plt

def double_exp(x, a, b, c, d):
    return a * np.exp(-x * b) + (d) * np.exp(-x * c)


def erfc_exp(t, A, tau, beta):
    return A * np.exp(-(t / tau) ** beta)


# def erfc_exp(t, a, koff, ks):
#     return (a 
#             * np.exp(( np.square(koff) * t) / ks) 
#             * erfc(np.sqrt((np.square(koff) * t) / ks))
#     )

# def streched_exp(t, a, tau, beta):
#     return a * np.exp(-(t / tau) ** beta)

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
    val: np.ndarray, t: np.ndarray,t_range, 
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

    if delete_nan:
        t, val = deleteNaN(val,t)
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


def create_fit_dataframe(data: pl.DataFrame,eq: callable) -> pl.DataFrame:

    fits = pl.DataFrame()
    for c in data.select("case").unique().to_series().to_list():
        case_df = data.filter(pl.col("case") == c)
        timestep = case_df.select("timestep").to_numpy().flatten()
        remaining = case_df.select("remaining").to_numpy().flatten()

        remaining = arr_minimize(remaining)
        t_range = np.log10(np.linspace(0,10_000,200) + 2)
        y_fit = value_fit(remaining, timestep, t_range=t_range, eq=eq)

        y_fit = 10**y_fit-1
        fits = fits.with_columns(
            pl.Series(y_fit).alias(f"{c}_{eq.__name__.replace('_exp','.exp')}")
        )



    return fits

def add_timestep(df:pl.DataFrame)->pl.DataFrame:
    return df.with_columns(
        pl.int_range(1,df.height+1).alias("timestep").add(1).log10()
    )

def melt(df:pl.DataFrame)->pl.DataFrame:
    return df.unpivot(index="timestep", variable_name="case", value_name="remaining")

def un_log(df:pl.DataFrame)->pl.DataFrame:
    return df.with_columns(
        (10**pl.col("timestep")).alias("timestep"),
        (10**pl.col("remaining")-1).alias("remaining"),
    )

def plot_fits(df,fits):
    return (
        ggplot()
        + geom_point(data=df, mapping=aes(x="timestep", y="remaining", color="case"))
        + scale_color_hue()
        + geom_line(data=fits, mapping=aes(x="timestep", y="remaining", color="case"))
        + scale_color_brewer()
        + scale_x_log10()
        + scale_y_log10()
        + labs(x="timestep", y="remaining")
        + theme_classic()
        + ggsize(width=1000, height=500)
        + scale_color_hue()

    )


def main():
    data_pp = pl.read_csv("data/dist_log10p1.csv")
    data_raw = pl.read_csv("data/duration_cont.csv")
    fit_tri = create_fit_dataframe(data_pp,tri_exp)
    fit_erfc = create_fit_dataframe(data_pp,erfc_exp)

    fitsm =melt(add_timestep(fit_tri)).with_columns(
        (10**pl.col("timestep")).alias("timestep"),
    )

    print(data_pp)
    print(fitsm)
    fig = plot_fits(un_log(data_pp),fitsm)
    fig.to_png("fits.png",dpi=300,w=10,h=5,unit="in")


if __name__ == "__main__":
    main()
    
