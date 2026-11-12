"""Figure 4 (ggrust).

One row per binding energy (2.8, 3.0, 3.5, 4.0 kT, 60 µM):

A (left column): snapshot of the clusters in the capsid, coloured by mean lifetime
B (right column): mean lifetime vs cluster size, three lifetime estimates
   (single-exponential fit, area under the survival curve, area + tail correction)

Lifetimes are recomputed from ``data/fulls/*.full.npy`` with ``diffuse_analysis``.
"""

import os

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from ggrust import (
    aes,
    element_blank,
    element_line,
    element_rect,
    element_text,
    geom_point,
    geom_smooth,
    geom_text,
    gggrid,
    ggplot,
    ggsave,
    ggsize,
    guide_colorbar,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_fill_gradient,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_classic,
)

import diffuse_analysis as da

KTS = ["2.80", "3.00", "3.50", "4.00"]
# snapshot shown in column A for each energy (the frames of the previous figure)
SNAPSHOTS = {"2.80": 17351, "3.00": 10773, "3.50": 20542, "4.00": 20371}

CAPSID = "#900C3F"
PURPLES = ("#fcfbfd", "#3f007d")  # matplotlib "Purples" ends
METHODS = {  # column -> (label, colour), as the old figure
    "tau_old": ("Single-exp fit", "#d62728"),
    "tau_int": ("Integral", "#1f77b4"),
    "tau_tail": ("Integral + tail", "#2ca02c"),
}
INK = "#1f1f1f"


def base_theme():
    """seaborn 'ticks' look, matching fig2.py."""
    return theme_classic() + theme(
        text=element_text(family="sans-serif", color=INK, size=17),
        axis_text=element_text(size=16, color=INK),
        axis_title=element_text(size=19, color=INK),
        axis_line=element_line(color=INK, size=0.8),
        axis_ticks=element_line(color=INK, size=0.8),
        axis_ticks_length=5,
        legend_background=element_blank(),
        legend_key=element_blank(),
        legend_title=element_blank(),
        legend_text=element_text(size=14),
        plot_background=element_rect(fill="white", color="white"),
        plot_tag=element_text(size=30, face="bold"),
        plot_tag_location="plot",
        plot_tag_position="topleft",
    )


def load(kt: str) -> pd.DataFrame:
    df = pd.read_csv(f"./data/12timepoint/{kt.replace('.', '')}.csv", index_col=False)
    return da.recompute_taus(df, f"./data/fulls/{kt}-60.full.npy", duration=300, primary="tail")


def panel_snapshot(df: pd.DataFrame, kt: str, capsid: pl.DataFrame, tag: str | None):
    """Clustered TFs at one frame inside the capsid outline, filled by mean lifetime."""
    clusters = df[df.clusterID > 0]
    frame = pl.from_pandas(clusters[clusters.timestep == SNAPSHOTS[kt]][["x", "y", "tau"]])
    note = pl.DataFrame(
        {
            "x": [-62.0, 62.0],
            "y": [-24.0, 24.0],
            "label": [f"t = {SNAPSHOTS[kt]}", f"{float(kt):.1f} kT"],
            "hjust": [0.0, 1.0],
        }
    )
    lo, hi = clusters.tau.min(), clusters.tau.max()

    return (
        ggplot()
        + geom_point(aes("x", "y"), data=capsid, color=CAPSID, size=2.2)
        + geom_point(
            aes("x", "y", fill="tau"), data=frame, shape=21, color="black", size=2.6, stroke=0.3
        )
        + geom_text(
            aes("x", "y", label="label", hjust="hjust"),
            data=note,
            size=15,
            color=INK,
            fontface="italic",
        )
        + scale_fill_gradient(
            low=PURPLES[0],
            high=PURPLES[1],
            limits=[lo, hi],
            name="Mean lifetime (a.u.)",
            guide=guide_colorbar(barheight=150, barwidth=12),
        )
        + scale_x_continuous(limits=[-92, 92])
        + scale_y_continuous(limits=[-31, 31])
        + labs(x="", y="", tag=tag)
        + base_theme()
        + theme(
            axis_line=element_blank(),
            axis_ticks=element_blank(),
            axis_text=element_blank(),
            axis_title=element_blank(),
            legend_position="left",
            legend_title=element_text(size=14, angle=90),
        )
    )


def panel_regression(df: pd.DataFrame, tag: str | None):
    """Mean lifetime vs cluster size with a linear fit per lifetime estimate."""
    simple = da.simplify(df[df.clusterID > 0])
    frames, notes = [], []
    for j, (col, (label, _)) in enumerate(METHODS.items()):
        m = simple[["clusterSize", col]].replace([np.inf, -np.inf], np.nan)
        m = m[m[col] > 0].dropna()
        r, _ = stats.pearsonr(m["clusterSize"], m[col])
        slope = np.polyfit(m["clusterSize"], m[col], 1)[0]
        frames.append(
            pl.DataFrame({"size": m["clusterSize"], "tau": m[col], "method": label})
        )
        notes.append((label, f"{label}: r = {r:.2f}, m = {slope:.1f}", j))
    data = pl.concat(frames)

    x0, x1 = data["size"].min(), data["size"].max()
    y1 = data["tau"].max()
    y0 = data["tau"].min()
    span = y1 - y0
    annot = pl.DataFrame(
        {
            "size": [x0] * len(notes),
            "tau": [y1 + span * (0.32 - 0.11 * j) for *_, j in notes],
            "label": [text for _, text, _ in notes],
            "method": [label for label, *_ in notes],
        }
    )
    n_frames = pl.DataFrame(
        {"size": [x1], "tau": [y0], "label": [f"N = {len(simple)}"]}
    )
    colors = {label: c for label, c in METHODS.values()}

    return (
        ggplot(data, aes("size", "tau", color="method"))
        + geom_point(size=2.4, alpha=0.45)
        + geom_smooth(aes(fill="method"), method="lm", se=True, size=1.6, alpha=0.18)
        + geom_text(
            aes(label="label"),
            data=annot,
            hjust=0,
            size=12,
            fontface="italic",
            show_legend=False,
        )
        + geom_text(
            aes("size", "tau", label="label"),
            data=n_frames,
            hjust=1,
            vjust=0,
            size=15,
            color=INK,
            fontface="italic",
            inherit_aes=False,
        )
        + scale_color_manual(values=colors, breaks=list(colors))
        + scale_fill_manual(values=colors, guide="none")
        + labs(x="Cluster size (#TF)", y="Mean lifetime (a.u.)", tag=tag)
        + base_theme()
        + theme(legend_position="none")
    )


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    capsid = pl.read_csv("./data/capsid2d.csv").select("x", "y")

    plots = []
    for row, kt in enumerate(KTS):
        df = load(kt)
        plots.append(panel_snapshot(df, kt, capsid, "A" if row == 0 else None))
        plots.append(panel_regression(df, "B" if row == 0 else None))

    fig = gggrid(plots, ncol=2, widths=[2.25, 1], align=False) + ggsize(1600, 1700)
    for ext in ("pdf", "png", "svg"):
        print(ggsave(fig, f"../Figures/fig4.{ext}"))


if __name__ == "__main__":
    main()
