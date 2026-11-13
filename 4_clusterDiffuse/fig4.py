"""Figure 4 (ggrust), one row per binding energy (2.8, 3.0, 3.5, 4.0 kT at 60 µM).

A (left column): snapshot of the clusters in the capsid, coloured by mean lifetime
   (one shared log colour scale for all rows)
B (right column): mean lifetime vs cluster size for three lifetime estimates
   (single-exponential fit, area under the survival curve, area + tail correction)

Lifetimes are recomputed from ``data/fulls/*.full.npy`` with ``diffuse_analysis``.
Same layout as the seaborn Fig. 4 in ``diffuse_analysis.py``, drawn with ggrust
(needs ggrust with stat_poly_eq "R"/"slope", infinite positions, gggrid
tag_level="unit" and the colorbar extend/minor ticks); writes Figures/legacy/.
"""

import os

import numpy as np
import pandas as pd
import polars as pl
from ggrust import (
    aes,
    annotate,
    coord_fixed,
    element_blank,
    element_line,
    element_rect,
    element_text,
    geom_path,
    geom_point,
    geom_smooth,
    gggrid,
    ggplot,
    ggsave,
    ggsize,
    guide_colorbar,
    guide_legend,
    labs,
    scale_color_manual,
    scale_fill_gradientn,
    scale_fill_manual,
    stat_poly_eq,
    theme,
    theme_classic,
    theme_void,
)

import diffuse_analysis as da

KTS = ["2.80", "3.00", "3.50", "4.00"]
# the frames shown in the published figure (the seaborn version picks them at random)
FRAMES = [16599, 5106, 20371, 20371]

CAPSID = "#900C3F"
# matplotlib "Purples" (the original panel A palette)
PURPLES = ["#fcfbfd", "#efedf5", "#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#54278f", "#3f007d"]
METHODS = {  # column -> (label, husl(3) colour)
    "tau_old": ("Exponential Decay", "#f77189"),
    "tau_int": ("Survival Integral", "#50b131"),
    "tau_tail": ("Tail-Corrected Integral", "#3ba3ec"),
}
COLORS = {label: color for label, color in METHODS.values()}
INK = "#1f1f1f"


def base_theme():
    """seaborn 'ticks' look, matching fig2.py."""
    return theme_classic() + theme(
        text=element_text(family="sans-serif", color=INK, size=15),
        axis_text=element_text(size=14, color=INK),
        axis_title=element_text(size=17, color=INK),
        axis_line=element_line(color=INK, size=0.7),
        axis_ticks=element_line(color=INK, size=0.7),
        axis_ticks_length=5,
        legend_background=element_blank(),
        legend_key=element_blank(),
        legend_text=element_text(size=14),
        legend_title=element_text(size=17),
        plot_background=element_rect(fill="white", color="white"),
        plot_tag=element_text(size=34, face="bold"),
        plot_tag_location="plot",
        plot_tag_position="topleft",
        legend_position="bottom",
    )


def load(kt: str) -> pd.DataFrame:
    df = pd.read_csv(f"./data/12timepoint/{kt.replace('.', '')}.csv", index_col=False)
    df = da.recompute_taus(df, f"./data/fulls/{kt}-60.full.npy", duration=300, primary="tail")
    return df[df.clusterID > 0]


def stadium(capsid: pd.DataFrame, n: int = 200) -> pl.DataFrame:
    """Capsid outline as one path: two straight walls joined by semicircles."""
    x, y = da.stadium_outline(capsid, n)
    return pl.DataFrame({"x": x, "y": y})


def snapshot(df: pd.DataFrame, t: int, capsid: pl.DataFrame, fill_scale):
    """Clusters of one frame inside the capsid, filled by mean lifetime."""
    frame = df[df.timestep == t]
    points = pl.DataFrame({"x": frame.x, "y": frame.y, "tau": frame.tau})
    return (
        ggplot()
        + geom_path(aes("x", "y"), data=capsid, color=CAPSID, size=1.6)
        + geom_point(aes("x", "y", fill="tau"), data=points, shape=21, color=INK, size=2.1, stroke=0.15)
        + annotate("text", x=-70, y=-27, label=f"t = {t}", hjust=0, size=8, fontface="italic", color=INK)
        + fill_scale
        + coord_fixed(ratio=1, xlim=[-92, 92], ylim=[-31, 31], expand=False)
        + base_theme()
        + theme_void()
        + theme(
            legend_position="bottom",
            # matplotlib colorbar: label under the bar, dark ticks
            legend_title_position="bottom",
            legend_title=element_text(size=17, hjust=0.5, color=INK),
            legend_text=element_text(size=14, color=INK),
            legend_ticks=element_line(color=INK, size=0.6),
        )
    )


def regression(df: pd.DataFrame):
    """Mean lifetime vs cluster size, linear fit, r and slope per lifetime estimate."""
    simple = da.simplify(df)
    frames = []
    for col, (label, _) in METHODS.items():
        m = simple[["clusterSize", col]].replace([np.inf, -np.inf], np.nan)
        m = m[m[col] > 0].dropna()
        frames.append(pl.DataFrame({"size": m.clusterSize, "tau": m[col], "method": label}))
    data = pl.concat(frames)
    n = len(np.unique(df.tau))  # as the seaborn figure

    return (
        ggplot(data, aes("size", "tau", color="method"))
        + geom_point(aes(fill="method"), shape=21, color="white", stroke=0.3, size=2.4, alpha=0.7)
        + geom_smooth(aes(fill="method"), method="lm", se=True, size=1.6, alpha=0.15)
        + stat_poly_eq(
            label=["R", "slope"],
            label_sep=", ",
            small_r=True,
            slope_name="m",
            slope_digits=1,
            label_x_npc="left",
            label_y_npc="top",
            size=7,
            fontface="bold_italic",
            show_legend=False,
        )
        + annotate("text", x=float("inf"), y=float("-inf"), label=f"N = {n}", hjust=1.3, vjust=-0.6, size=10, fontface="italic", color=INK)
        + scale_color_manual(values=COLORS, breaks=list(COLORS), guide=guide_legend(ncol=1))
        + scale_fill_manual(values=COLORS, breaks=list(COLORS), guide=guide_legend(ncol=1))
        + labs(x="Cluster Size (#TF)", y="Mean Lifetime (a.u.)", color="", fill="")
        + base_theme()
        + theme(legend_title=element_blank())
    )


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    capsid = stadium(pd.read_csv("./data/capsid2d.csv"))
    dfs = {kt: load(kt) for kt in KTS}

    taus = pd.concat([d.tau for d in dfs.values()])
    fill_scale = scale_fill_gradientn(
        colors=PURPLES,
        trans="log10",
        limits=[taus[taus > 0].min(), taus.max()],
        breaks=[5, 10, 20, 50, 100, 200, 500],
        name="Mean Lifetime (a.u.)",
        guide=guide_colorbar(
            barwidth=330,
            barheight=12,
            display="gradient",
            shape="triangle",  # matplotlib extend="both"
            minor_ticks=True,
        ),
    )

    snapshots = gggrid(
        [snapshot(dfs[kt], t, capsid, fill_scale) for kt, t in zip(KTS, FRAMES)],
        ncol=1,
        guides="collect",
        tag_level="unit",
    )
    regressions = gggrid(
        [regression(dfs[kt]) for kt in KTS],
        ncol=1,
        guides="collect",
        tag_level="unit",
    )
    fig = gggrid([snapshots, regressions], ncol=2, widths=[1.6, 1], tag_levels="A") + ggsize(1400, 1560)
    fig += theme(plot_tag=element_text(size=34, face="bold", color=INK))

    # alternative version; the primary Fig. 4 is the seaborn one (diffuse_analysis.py)
    for ext in ("pdf", "png"):
        print(ggsave(fig, f"../Figures/legacy/fig4_ggrust.{ext}"))


if __name__ == "__main__":
    main()
