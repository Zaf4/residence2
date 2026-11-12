"""Figure 2 (ggrust).

A: NS residence-time distributions at 60 µM (2.8-4.0 kT, prof's data in
   data/distributions/), each fitted with its effective dimension (number of exponentials)
B: effective dimension vs concentration for non-specific (NS) and specific (SP) sites
C: effective dimension vs energy at 60 µM (NS and SP)

The full fit grid and the residual boxplots (old Fig. 2A/B) live in ``legacy/``
and go to the SI.
"""

import os

import numpy as np
import polars as pl
from scipy.optimize import least_squares
from scipy.special import logsumexp
from ggrust import (
    aes,
    element_blank,
    element_line,
    element_rect,
    element_text,
    facet_wrap,
    geom_label,
    geom_line,
    geom_point,
    gggrid,
    ggplot,
    ggsave,
    ggsize,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
    scale_x_log10,
    scale_y_log10,
    theme,
    theme_classic,
)

# panel A: every 60 µM case with both a distribution and an effective dimension
UM = "60"
KTS_A = ["2.80", "3.00", "3.50", "4.00"]
SITE_A = "NS"

# palettes taken from the original seaborn figure
# husl(4), the fit-line colours of the old Fig. 2A
FIT_COLORS = dict(
    zip(["2.8 kT", "3.0 kT", "3.5 kT", "4.0 kT"], ["#f77189", "#97a431", "#36ada4", "#a48cf4"])
)


def tint(hex_color: str, white: float = 0.75) -> str:
    """Mix a colour with white (white=1 gives pure white)."""
    rgb = np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)])
    return "#" + "".join(f"{round(c + (255 - c) * white):02x}" for c in rgb)


# whitish tints of the fit colours for the panel A markers
POINT_FILLS = {kt: tint(c) for kt, c in FIT_COLORS.items()}
KT_COLORS = {  # magma(5), middle three, darker = stronger binding
    "2.8 kT": "#f1605d",
    "3.0 kT": "#b73779",
    "3.5 kT": "#721f81",
}
SITE_COLORS = {"NS": "#382a54", "SP": "#60ceac"}  # mako_r ends
SITE_LABELS = {
    "NS": "Nonspecific interactions",
    "SP": "Specific interactions",
}
INK = "#1f1f1f"


def base_theme():
    """seaborn 'ticks' look: no top/right spines, no grid, no minor ticks, no legend frame."""
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
        legend_text=element_text(size=16),
        strip_background=element_blank(),
        strip_text=element_text(size=17, face="bold"),
        plot_title=element_text(size=17, face="bold"),
        plot_background=element_rect(fill="white", color="white"),
        plot_tag=element_text(size=26, face="bold"),
        plot_tag_location="plot",  # tags at the plot corner, so A, B, C line up
        plot_tag_position="topleft",
    )


def sample_ends_favored(values: np.ndarray, n: int = 10, frac: float = 0.2) -> np.ndarray:
    """Indices of a sample that keeps both ends and thins the crowded middle.

    Same weighting as the original seaborn figure.
    """
    idx = np.flatnonzero(~np.isnan(values))
    ts = idx + 1.0
    tmax = ts.max()
    weights = 10 ** (np.abs(ts - tmax / 2) / tmax) ** 2
    rng = np.random.default_rng(42)
    mid = rng.choice(idx, size=int(len(idx) * frac), replace=False, p=weights / weights.sum())
    return np.unique(np.concatenate([idx[:n], mid, idx[-n:]]))


def effective_dimensions() -> pl.DataFrame:
    """Effective dimension at 60 µM for every energy and site (prof's dim_vs_energy.csv)."""
    return (
        pl.read_csv("./data/dimensions/dim_vs_energy.csv")
        .rename({"energy [kT]": "energy"})
        .unpivot(index="energy", variable_name="site", value_name="dim")
        .with_columns(site=pl.col("site").str.extract(r"\[(\w+),"))
    )


def load_distribution(site: str, kt: str) -> tuple[np.ndarray, np.ndarray]:
    """Residence-time distribution of one case, one point per distinct count.

    Repeated counts (the long tail of 1s, 2s, ...) are collapsed to their median
    duration, as ``durations_minimized.csv`` did for the old figure.
    """
    df = (
        pl.read_csv(f"./data/distributions/csv/{site}_distributions.tidy.csv")
        .filter(pl.col("case_id") == f"{kt}_{UM}_{site.lower()}")
        .group_by("count")
        .agg(pl.col("duration").median())
        .sort("duration")
    )
    return df["duration"].to_numpy().astype(float), df["count"].to_numpy().astype(float)


def log_n_exp(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """log of sum_i a_i exp(-k_i t), with q = [log a_1..n, log k_1..n]."""
    n = len(q) // 2
    return logsumexp(q[:n, None] - np.outer(np.exp(q[n:]), t), axis=0)


def fit_n_exp(t: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Least-squares fit of n exponentials in log space (every decade counts),
    best of a few starting rate spreads."""
    best = None
    for k_fast in (-1, 0, 0.5):
        for k_slow in (-2.5, -3, -3.5):
            q0 = np.r_[np.log(np.full(n, y[0] / n)), np.log(np.logspace(k_fast, k_slow, n))]
            res = least_squares(lambda q: log_n_exp(t, q) - np.log(y), q0)
            if best is None or res.cost < best.cost:
                best = res
    return best.x


def effective_fit(t: np.ndarray, y: np.ndarray, d_eff: float) -> np.ndarray:
    """Fit with a non-integer number of exponentials d_eff.

    d_eff is an average over fits, so mix the floor(d)- and ceil(d)-exponential
    fits so that their mean exponent count is d_eff
    (3.145 -> 85.5 % of the 3-exponential fit + 14.5 % of the 4-exponential fit).
    """
    lo, w = int(np.floor(d_eff)), d_eff - np.floor(d_eff)
    grid = np.arange(1, t.max() + 1)
    fit_lo = np.exp(log_n_exp(grid, fit_n_exp(t, y, lo)))
    fit_hi = np.exp(log_n_exp(grid, fit_n_exp(t, y, lo + 1)))
    return (1 - w) * fit_lo + w * fit_hi


def panel_fit() -> "ggplot":
    """NS distributions at 60 µM, each fitted with its effective dimension.

    Styled after the old Fig. 2A: white markers with black edges, thick husl fit lines.
    """
    dims = effective_dimensions().filter(pl.col("site") == SITE_A)
    point_frames, fit_frames = [], []
    for kt in KTS_A:
        t, y = load_distribution(SITE_A, kt)
        d_eff = dims.filter(pl.col("energy") == float(kt))["dim"].item()
        energy = f"{float(kt):.1f} kT"

        # points: thin the crowded middle, keep both ends (as the old figure)
        dense = np.full(int(t.max()), np.nan)
        dense[t.astype(int) - 1] = y
        keep = sample_ends_favored(dense, n=8, frac=0.15)
        point_frames.append(
            pl.DataFrame({"duration": keep + 1, "count": dense[keep], "energy": energy})
        )
        fit_frames.append(
            pl.DataFrame(
                {
                    "duration": np.arange(1, t.max() + 1),
                    "count": effective_fit(t, y, d_eff),
                    "energy": energy,
                }
            ).filter(pl.col("count") >= 1)  # below one event is off the y-axis
        )
    points, fits = pl.concat(point_frames), pl.concat(fit_frames)

    return (
        ggplot()
        + geom_point(
            aes("duration", "count", fill="energy"),
            data=points,
            shape=21,
            color="black",
            size=5.5,
            stroke=0.5,
        )
        + geom_line(aes("duration", "count", color="energy"), data=fits, size=2.6)
        + scale_color_manual(values=FIT_COLORS)
        + scale_fill_manual(values=POINT_FILLS)
        + scale_x_log10(breaks=[1, 10, 100, 1000], format=".0e", limits=[0.72, 3.85e3])
        + scale_y_log10(breaks=[1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], format=".0e", limits=[0.5, 1.2e7])
        + labs(x="Duration (a.u.)", y="Occurence (n)", title=" ")  # blank title = room of B's strips
        + base_theme()
        + theme(
            exponent_format=("pow", 0, 0),
            legend_position=[0.98, 0.98],
            legend_justification=[1, 1],
        )
    )


def panel_sites() -> "ggplot":
    frames = []
    for site in ["NS", "SP"]:
        df = pl.read_csv(f"./data/dimensions/{site.lower()}_dimensions.csv")
        df = df.unpivot(index="concentration [uM]", variable_name="energy", value_name="dim")
        frames.append(df.with_columns(site=pl.lit(SITE_LABELS[site])))
    data = pl.concat(frames).with_columns(
        energy=pl.col("energy").str.extract(r"\[(.+)\]"),
        concentration=pl.col("concentration [uM]"),
    )

    return (
        ggplot(data, aes("concentration", "dim", color="energy"))
        + geom_line(size=1.6)
        + geom_point(aes(fill="energy"), shape=21, color=INK, size=5, stroke=0.6)
        + facet_wrap("site", ncol=2)
        + scale_color_manual(values=KT_COLORS)
        + scale_fill_manual(values=KT_COLORS)
        + scale_x_continuous(breaks=[10, 20, 40, 60, 80, 120])
        + labs(x="Concentration (µM)", y="Effective dimension")
        + base_theme()
        + theme(legend_position=[0.02, 0.98], legend_justification=[0, 1])
    )


def panel_energy() -> "ggplot":
    data = effective_dimensions()
    tag = pl.DataFrame({"energy": [4.5], "dim": [3.2], "label": [f"{UM} µM"]})

    return (
        ggplot(data, aes("energy", "dim", color="site"))
        + geom_line(size=1.6)
        + geom_point(aes(fill="site"), shape=21, color=INK, size=5, stroke=0.6)
        + geom_label(
            aes(label="label"),
            data=tag,
            hjust=1,
            vjust=0.5,
            size=9,
            color=INK,
            fill="white",
            label_r=0.35,
            label_padding=0.4,
            label_size=0.8,
            show_legend=False,
        )
        + scale_color_manual(
            values=SITE_COLORS, breaks=list(SITE_LABELS), labels=list(SITE_LABELS.values())
        )
        + scale_fill_manual(
            values=SITE_COLORS, breaks=list(SITE_LABELS), labels=list(SITE_LABELS.values())
        )
        + scale_x_continuous(breaks=[2.8, 3.0, 3.2, 3.5, 4.0, 4.5])
        + labs(x="Energy (kT)", y="Effective dimension", title=" ")
        + base_theme()
        + theme(legend_position=[0.02, 0.98], legend_justification=[0, 1])
    )


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    fig = gggrid(
        [panel_fit(), panel_sites(), panel_energy()],
        widths=[1, 1.7, 1],
        ncol=3,
        tag_levels="A",
        align=False,
    ) + ggsize(1500, 430)

    for ext in ("pdf", "png", "svg"):
        print(ggsave(fig, f"../Figures/fig2.{ext}"))


if __name__ == "__main__":
    main()
