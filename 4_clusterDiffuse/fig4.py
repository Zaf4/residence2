"""Figure 4 (ggrust), one row per binding energy (2.8, 3.0, 3.5, 4.0 kT at 60 µM).

A (left column): snapshot of the clusters in the capsid, coloured by mean lifetime
   (one shared log colour scale for all rows)
B (right column): mean lifetime vs cluster size for three lifetime estimates
   (single-exponential fit, area under the survival curve, area + tail correction)

Lifetimes are recomputed from ``data/fulls/*.full.npy`` with ``diffuse_analysis``.
Alternative to the primary seaborn Fig. 4 in ``diffuse_analysis.py``; writes Figures/legacy/.
"""

import os

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from ggrust import (
    aes,
    coord_fixed,
    element_blank,
    element_line,
    element_rect,
    element_text,
    facet_wrap,
    geom_path,
    geom_point,
    geom_smooth,
    geom_text,
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
    theme,
    theme_classic,
)

import diffuse_analysis as da

KTS = ["2.80", "3.00", "3.50", "4.00"]
ENERGIES = [f"{float(kt):.1f} kT" for kt in KTS]

CAPSID = "#900C3F"
# matplotlib "Purples" (the original panel A palette)
PURPLES = ["#fcfbfd", "#efedf5", "#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#54278f", "#3f007d"]
METHODS = {  # column -> (label, husl(3) colour)
    "tau_old": ("Exponential Decay", "#f77189"),
    "tau_int": ("Survival Integral", "#50b131"),
    "tau_tail": ("Tail-Corrected Integral", "#3ba3ec"),
}
INK = "#1f1f1f"


def base_theme():
    """seaborn 'ticks' look, matching fig2.py."""
    return theme_classic() + theme(
        text=element_text(family="sans-serif", color=INK, size=15),
        axis_text=element_text(size=14, color=INK),
        axis_title=element_text(size=17, color=INK),
        axis_line=element_line(color=INK, size=0.7),
        axis_ticks=element_line(color=INK, size=0.7),
        axis_ticks_length=4,
        legend_background=element_blank(),
        legend_key=element_blank(),
        legend_text=element_text(size=14),
        legend_title=element_text(size=15),
        strip_background=element_blank(),
        strip_text=element_text(size=16, face="bold"),
        plot_background=element_rect(fill="white", color="white"),
        plot_tag=element_text(size=30, face="bold"),
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
    radius = capsid.y.abs().max()
    half = capsid.x.abs().max() - radius
    right = np.linspace(-np.pi / 2, np.pi / 2, n)
    left = right + np.pi
    x = np.r_[half + radius * np.cos(right), -half + radius * np.cos(left), half]
    y = np.r_[radius * np.sin(right), radius * np.sin(left), -radius]
    return pl.DataFrame({"x": x, "y": y})


def panel_snapshots(dfs: dict[str, pd.DataFrame], capsid: pl.DataFrame):
    """One random frame per energy, clusters filled by mean lifetime, shared log scale."""
    rng = np.random.default_rng()
    frames = []
    for kt, energy in zip(KTS, ENERGIES):
        df = dfs[kt]
        t = rng.choice(df.timestep.unique())
        f = df[df.timestep == t]
        # frame time goes in the strip, where clusters can never cover it
        frames.append(
            pl.DataFrame({"x": f.x, "y": f.y, "tau": f.tau, "energy": f"{energy}  ·  t = {t}"})
        )
    points = pl.concat(frames)
    taus = pd.concat([d.tau for d in dfs.values()])

    return (
        ggplot()
        + geom_path(aes("x", "y"), data=capsid, color=CAPSID, size=1.6)
        + geom_point(
            aes("x", "y", fill="tau"), data=points, shape=21, color=INK, size=2.1, stroke=0.15
        )
        + facet_wrap("energy", ncol=1, order=1)
        + scale_fill_gradientn(
            colors=PURPLES,
            trans="log10",
            limits=[taus[taus > 0].min(), taus.max()],
            breaks=[5, 10, 20, 50, 100, 200, 500],
            name="Mean Lifetime (a.u.)",
            guide=guide_colorbar(barwidth=300, barheight=10),
        )
        + coord_fixed(ratio=1, xlim=[-92, 92], ylim=[-31, 31])
        + labs(x="", y="")
        + base_theme()
        + theme(
            axis_line=element_blank(),
            axis_ticks=element_blank(),
            axis_text=element_blank(),
            axis_title=element_blank(),
        )
    )


def panel_regression(dfs: dict[str, pd.DataFrame]):
    """Mean lifetime vs cluster size, linear fit per lifetime estimate, free scales."""
    points, notes, counts = [], [], []
    for kt, energy in zip(KTS, ENERGIES):
        simple = da.simplify(dfs[kt])
        frames = []
        for col, (label, _) in METHODS.items():
            m = simple[["clusterSize", col]].replace([np.inf, -np.inf], np.nan)
            m = m[m[col] > 0].dropna()
            frames.append(
                pl.DataFrame({"size": m.clusterSize, "tau": m[col], "method": label, "energy": energy})
            )
        data = pl.concat(frames)
        points.append(data)

        # r and slope per method, stacked in the headroom above the data
        x0, x1 = data["size"].min(), data["size"].max()
        y0, y1 = data["tau"].min(), data["tau"].max()
        span = y1 - y0
        for j, (label, _) in enumerate(METHODS.values()):
            sub = data.filter(pl.col("method") == label)
            r, _ = stats.pearsonr(sub["size"], sub["tau"])
            slope = np.polyfit(sub["size"], sub["tau"], 1)[0]
            notes.append(
                {
                    "size": x0,
                    "tau": y1 + span * (0.36 - 0.11 * j),
                    "label": f"r = {r:.2f}, m = {slope:.1f}",
                    "method": label,
                    "energy": energy,
                }
            )
        counts.append({"size": x1, "tau": y0, "label": f"N = {len(simple)}", "energy": energy})
    data = pl.concat(points)
    colors = {label: c for label, c in METHODS.values()}

    return (
        ggplot(data, aes("size", "tau", color="method"))
        + geom_point(size=2.2, alpha=0.6)
        + geom_smooth(aes(fill="method"), method="lm", se=True, size=1.4, alpha=0.15)
        + geom_text(
            aes(label="label"),
            data=pl.DataFrame(notes),
            hjust=0,
            size=9,
            fontface="bold_italic",
            show_legend=False,
        )
        + geom_text(
            aes("size", "tau", label="label"),
            data=pl.DataFrame(counts),
            hjust=1,
            vjust=0,
            size=10,
            color=INK,
            fontface="italic",
            inherit_aes=False,
        )
        + facet_wrap("energy", ncol=1, scales="free", order=1)
        + scale_color_manual(values=colors, breaks=list(colors), guide=guide_legend(ncol=1))
        + scale_fill_manual(values=colors, guide="none")
        + labs(x="Cluster Size (#TF)", y="Mean Lifetime (a.u.)")
        + base_theme()
        + theme(legend_title=element_blank())
    )


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    capsid = stadium(pd.read_csv("./data/capsid2d.csv"))
    dfs = {kt: load(kt) for kt in KTS}

    fig = gggrid(
        [panel_snapshots(dfs, capsid), panel_regression(dfs)],
        ncol=2,
        widths=[1.25, 1],
        tag_levels="A",
        align=True,
    ) + ggsize(1500, 1380)

    # alternative version; the primary Fig. 4 is the seaborn one (diffuse_analysis.py)
    for ext in ("pdf", "png"):
        print(ggsave(fig, f"../Figures/legacy/fig4_ggrust.{ext}"))


if __name__ == "__main__":
    main()
