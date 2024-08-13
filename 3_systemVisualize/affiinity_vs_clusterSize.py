from lets_plot import *
import numpy as np
import pandas as pd
import os
import polars as pl
from system_visualize import data2df, cluster_single_df


KTS = ["2.80", "3.00", "3.50", "4.00"]
UMS = ["10", "20", "40", "60"]


def extract_counts(kt: str, um: int) -> pl.DataFrame:
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
    print(f"{kt=}, {um=}")
    coor = data2df(f"./data_extra/{kt}_{um}.extra")  # read as the dataframe
    df_cluster = cluster_single_df(coor)  # find clusters in the dataframe
    df = pl.from_pandas(df_cluster)  # convert to polars dataframe
    df = df.with_columns(
        pl.col("clusterID").fill_null(0).cast(pl.Int32),
        pl.col(["type", "atomID"]).cast(pl.Int32),
    )
    df = df.filter(pl.col("type") == 5, pl.col("clusterID") > 0)
    counts = (
        df.group_by("clusterID")
        .agg(pl.col("clusterID").count().alias("size"))
        .sort("size")
        .with_columns(pl.lit(f"{kt}").alias("kT"), pl.lit(um).alias("um"))
    )

    return counts


def collect_dataframes(KTS: list, UMS: list) -> pl.DataFrame:
    """
    Collect dataframes for each kT and um
    """
    df_all = pl.DataFrame()
    for kt in KTS:
        for um in UMS:
            counts = extract_counts(kt=kt, um=um)
            df_all = pl.concat([df_all, counts])
    return df_all


def cluster_size_vs_affinity(
    df: pl.DataFrame,
    group="um",
    X=KTS,
    ymin="confidence_95_low",
    ymax="confidence_95_high",
):
    """

    Parameters
    ----------
    df : pl.DataFrame
        dataframe to visualize.
        with cluster sizes, affinities and concentrations.
    """
    df = df.sort(group)

    color_kwargs = {
        "palette": "Reds",
        "labels": [f"{x}µM" for x in UMS],
        "direction": 1,
    }

    graph = (
        ggplot(df, aes(x="kT", y="mean_cluster_size"))
        + geom_line(aes(color=as_discrete(group)))
        + geom_errorbar(aes(color=as_discrete(group), ymin=ymin, ymax=ymax))
        + geom_point(
            shape=21, mapping=aes(fill=as_discrete(group)), color="#1f1f1f", size=4
        )
        + scale_color_brewer(**color_kwargs)
        + scale_fill_brewer(**color_kwargs)
        + theme_classic()
        + scale_x_continuous(breaks=[float(x) for x in X])
        + theme(
            legend_position="top",
            legend_title=element_blank(),
            legend_text=element_text(size=12),
            panel_background=element_blank(),
            plot_background=element_blank(),
            legend_background=element_blank(),

        )
    ) + labs(title="", x="Affinity (kT)", y="Cluster size")

    return graph + ggsize(800, 400)


def cluster_size_vs_concentration(
    df: pl.DataFrame,
    group="kT",
    X=UMS,
    ymin="confidence_95_low",
    ymax="confidence_95_high",
):
    """

    Parameters
    ----------
    df : pl.DataFrame
        dataframe to visualize.
        with cluster sizes, affinities and concentrations.
    """

    df = df.sort(group)

    color_kwargs = {
        "palette": "Purples",
        "labels": [f"{float(x):.1f}kT" for x in KTS],
        "direction": 1,
    }

    graph = (
        ggplot(
            df,
            aes(
                x="um",
                y="mean_cluster_size",
            ),
        )
        + geom_line(aes(color=as_discrete(group)))
        + geom_errorbar(aes(color=as_discrete(group), ymin=ymin, ymax=ymax))
        + geom_point(
            shape=21, mapping=aes(fill=as_discrete(group)), color="#1f1f1f", size=4
        )
        + scale_color_brewer(**color_kwargs)
        + scale_fill_brewer(**color_kwargs)
        + labs(title="", x="Concentration (µM)", y="Cluster size")
        + theme_classic()
        + scale_x_continuous(breaks=[int(x) for x in X])
        + theme(
            legend_position="top",
            legend_title=element_blank(),
            legend_text=element_text(size=12),
            panel_background=element_blank(),
            plot_background=element_blank(),
            legend_background=element_blank(),
        )
    )

    return graph + ggsize(800, 400)


def process_df(df: pl.DataFrame) -> pl.DataFrame:
    grouped = (
        df.group_by(["kT", "um"])
        .agg(
            pl.col("size").mean().alias("mean_cluster_size"),
            pl.col("size").std().alias("std_cluster_size"),
            pl.col("size").count().alias("sample_size"),
        )
        .with_columns(
            (pl.col("mean_cluster_size") - pl.col("std_cluster_size")).alias(
                "cluster_size_minus_std"
            ),
            (pl.col("mean_cluster_size") + pl.col("std_cluster_size")).alias(
                "cluster_size_plus_std"
            ),
            (
                pl.col("mean_cluster_size")
                - 1.96 * (pl.col("std_cluster_size") / pl.col("sample_size"))
            ).alias("confidence_95_low"),  # z = 1.96 for CI 95
            (
                pl.col("mean_cluster_size")
                + 1.96 * (pl.col("std_cluster_size") / pl.col("sample_size"))
            ).alias("confidence_95_high"),
        )
        .with_columns(
            pl.col("kT").cast(pl.Float64),
            pl.col("um").cast(pl.Float32),
        )
    )
    return grouped


def main():
    if not os.path.exists("data_extra/collected.parquet"):
        df = collect_dataframes(KTS, UMS)
        df.write_parquet("data_extra/collected.parquet")
    else:
        df = pl.read_parquet("data_extra/collected.parquet")

    grouped = process_df(df)
    print(grouped.sample(10))

    ymin = "confidence_95_low"
    ymax = "confidence_95_high"
    affinity_graph = cluster_size_vs_affinity(grouped, ymin=ymin, ymax=ymax)
    concentration_graph = cluster_size_vs_concentration(grouped, ymin=ymin, ymax=ymax)
    ax = (
        gggrid([affinity_graph, concentration_graph])
        + ggsize(700, 300)
        + theme(panel_background=element_blank(), plot_background=element_blank())
    )
    # ggsave(ax, "../Figures/fig3DE.html", path=".")
    ggsave(ax, "../Figures/fig3DE.pdf", path=".")
    ggsave(ax, "../Figures/fig3DE.png", path=".")
    # ggsave(affinity_graph, "../Figures/fig3DE.png", path=".")

    return


if __name__ == "__main__":
    main()
