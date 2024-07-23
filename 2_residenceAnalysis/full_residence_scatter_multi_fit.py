import os

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.progress import track
import warnings
warnings.filterwarnings("ignore", message="The palette list has more values")


# suppress the warning
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



    df = df.dropna(axis=0)

    # weight calculation to prevent overcrowding
    ts = np.array(df.timestep)
    # # weights = ((1/(ts*ts[::-1]))*10**5)**2
    tmax = df.timestep.max()
    weights = np.abs(ts - tmax / 2)
    weights = 10 ** (weights / tmax) ** 2
    sampled_df = pd.concat(
        [df.head(n), df.sample(frac=0.20, random_state=42,weights=weights), df.tail(n)]
    )
    return sampled_df


def scatterit_multi_func(
    df: pd.DataFrame, fits_list: list[pd.DataFrame], i: int, j: int, axes, palette: str, **kwargs
) -> None:
    """generates scatter plot with fits from multiple functions

    Parameters
    ----------
    df : pd.DataFrame
        Distribution data
    fits : pd.DataFrame
        Fit data
    i : int
        row of axes
    j : int
        column of axes
    axes : plt.Axes
        _description_
    palette : str
        color palette for the plots
    """

    # preprocess raw data and fits for graph------------------------------------
    cols = [col for col in df]
    df["timestep"] = np.arange(len(df)) + 1

    df = pd.melt(
        df, id_vars=["timestep"], value_vars=cols, var_name="case", value_name="value"
    )

    # df.dropna(axis=0,inplace=True)

    ax = axes[i, j]

    # font settings
    font = {
        "family": "Sans Serif",
        "weight": "light",
        "size": 20,
    }

    # weight calculation to prevent overcrowding
    # ts = np.array(df.timestep)
    # # weights = ((1/(ts*ts[::-1]))*10**5)**2
    # tmax = df.timestep.max()
    # weights = np.abs(ts - tmax / 2)
    # weights = 10 ** (weights / tmax) ** 2

    # sampled data
    # sampled_df = df.sample(frac=0.4, random_state=42, weights=weights)
    sampled_df = sample_ends_favored(df, n=10)
    # scattter plot (Data)------------------------------------------------------
    sns.scatterplot(
        data=sampled_df,
        x="timestep",
        y="value",
        #palette=palette,
        #hue="case",
        #hue_order=cols,
        color="white",
        alpha=1,
        s=300,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
        **kwargs,
    )

    # Create a darker version of the viridis palette

    # viridis_palette = sns.color_palette(
    #     palette, as_cmap=True
    # )  # Get the original viridis color palette

    # Create a darker version of the viridis palette
    # darker = sns.color_palette(
    #     [tuple([min(1, c + 0.2) for c in color]) for color in viridis_palette.colors]
    # )
    #darker_palette = sns.color_palette(darker, as_cmap=True)[40::50]

    annotated_fit_list = list()
    for fits in fits_list:
        # Line plot (Fits)----------------------------------------------------------
        fits["timestep"] = np.arange(len(df)) + 1
        fits = pd.melt(
        fits, id_vars=["timestep","equation"], value_vars=cols, var_name="case", value_name="value"
        )
        annotated_fit_list.append(fits)

    fits = pd.concat(annotated_fit_list, axis=0, ignore_index=True)
    print(fits.head())

    sns.lineplot(
        data=fits,
        x="timestep",
        y="value",
        palette="husl",
        hue="equation",
        ax=ax,
        linewidth=4.5,
        linestyle="solid",
        alpha=1,
        **kwargs,
    )

    # graph settings------------------------------------------------------------

    ax.tick_params(axis="both", labelsize=21)
    #ax.get_legend().remove()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim([0.72, 3.85e3])
    ax.set_ylim([0.5, 1.15e6])
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    # ax.set_xlabel('Duration (a.u.)', fontdict=font)
    # ax.set_ylabel('Occurence', fontdict=font)

    return


def generate_fit_graph_multi_func(
    datafile: os.PathLike = "./data/durations_minimized.csv",
    concentrations: list[str] = ["10", "20", "40", "60"],
    kts: list[str] = ["1.00", "2.80", "3.00", "3.50", "4.00"],
    figname: str = "noname",
) -> mpl.figure.Figure:
    
    durations = pd.read_csv(datafile, index_col=None)
    # eqnames = ["ED", "DED", "TED", "QED", "PED", "Powerlaw", "ERFC"]
    eqnames = ['ED','DED','PED','Powerlaw'] #for smaller figures

    nrow, ncol = len(kts), len(concentrations)

    # seaborn settings
    sns.set_theme(
        style="ticks",
        rc={
            #'font.weight': 'bold',
            "font.family": "sans-serif",
            "axes.spines.top": "False",
            "axes.spines.right": "False",
            "ytick.minor.size": "0",
            "xtick.minor.size": "0",
            "ytick.major.size": "5",
            "xtick.major.size": "5",
            "legend.frameon": False,
        },
    )

    fig, axes = plt.subplots(
    nrow, ncol, sharex=True, sharey=True, figsize=(ncol * 4, nrow * 3)
    )

    for i, kt in track(enumerate(kts)):
        for j, conc in enumerate(concentrations):
            

            # columns wtih the keyword
            cols = [x for x in durations if kt in x and conc in x]
            print(f"Concentration: {conc}, kT: {kt}, Columns: {cols}")
            
            partial_data = durations[cols]
            # collecting fit dataframes
            fit_list = list()
            for eqname in eqnames:
                fitfile = f"./data/{eqname}.csv"
                fits = pd.read_csv(fitfile, index_col=None)
                partial_fits = fits[cols]
                partial_fits["equation"] = eqname
                fit_list.append(partial_fits)

            scatterit_multi_func(
                partial_data, fit_list, axes=axes, palette="magma", i=i, j=j
            )

                # legend = [f'{x[:4]}kT' for x in cols]

            # fig.legend(legend,loc=(0.20,0.92),fontsize=15,markerscale=1.4,
            #    labelspacing=0.25,edgecolor='k')
            fig.supxlabel("Duration (a.u.)", fontsize=24, fontweight="light")
            fig.supylabel("Occurence (n)", fontsize=24, fontweight="light")
            plt.tight_layout()
            plt.savefig(f"../Figures/{figname}.pdf", transparent=True, bbox_inches="tight")



    return fig

if __name__ == "__main__":
    # changing working directory to current directory name
    os.chdir(os.path.dirname(__file__))

    ums = ["10", "60"]
    kts = [ "2.80", "3.50", "4.00"]
    
    generate_fit_graph_multi_func(concentrations=ums, kts=kts, figname = "fig2A")
