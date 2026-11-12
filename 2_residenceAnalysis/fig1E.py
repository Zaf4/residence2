"""Figure 1E, redrawn from data and spliced into the Illustrator-made Figure 1.

Panel E shows how a residence-time distribution becomes an effective dimension:
raw distribution -> median per distinct count -> fit with a sum of exponentials,
whose (non-integer) number of terms is the effective dimension d_eff.

Data: NS sites, 3.5 kT, 60 µM (d_eff = 3.145, prof's dim_vs_energy.csv).

Panels A-D stay exactly as drawn in Illustrator: the original page
(Figures/legacy/fig1_original.pdf) is copied, the old panel E is covered with
white, and the new panel E is placed on top as vector PDF (PyMuPDF).
"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pymupdf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig2 import (  # noqa: E402  (same data loading and fitting as Fig. 2A)
    effective_dimensions,
    effective_fit,
    fit_n_exp,
)
import polars as pl  # noqa: E402

SITE, KT, UM = "NS", "3.50", "60"
ORIGINAL = "../Figures/legacy/fig1_original.pdf"
PANEL = "../Figures/fig1E.pdf"
OUTPUT = "../Figures/fig1.pdf"

# region of panel E on the 408 x 301 pt page, from the original's layout:
# below panel C's "Time (a.u.)" label, right of the "E" tag
E_RECT = pymupdf.Rect(16, 174, 408.4, 301.1)
TAG_GAP = pymupdf.Rect(0, 193, 16, 301.1)  # under the "E" tag, left of E_RECT

RED = "#ed1c24"  # Fig. 1 red
INK = "#231f20"
COMPONENT_COLORS = ["#f77189", "#50b131", "#3ba3ec", "#a48cf4"]  # husl, as Fig. 2/4


def load_raw() -> tuple[np.ndarray, np.ndarray]:
    df = pl.read_csv(f"./data/distributions/csv/{SITE}_distributions.tidy.csv").filter(
        pl.col("case_id") == f"{KT}_{UM}_{SITE.lower()}"
    )
    return df["duration"].to_numpy().astype(float), df["count"].to_numpy().astype(float)


def median_per_count(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One point per distinct count, at the median duration (as Fig. 2A)."""
    df = (
        pl.DataFrame({"duration": t, "count": y})
        .group_by("count")
        .agg(pl.col("duration").median())
        .sort("duration")
    )
    return df["duration"].to_numpy(), df["count"].to_numpy()


def representative(t: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n real points, the nearest ones to n evenly spaced log-durations."""
    targets = np.logspace(0, np.log10(t.max()), n)
    idx = np.unique([np.abs(np.log(t) - np.log(x)).argmin() for x in targets])
    return t[idx], y[idx]


def style_axes(ax, xlabel: str = "Duration", ylabel: str | None = None) -> None:
    """Schematic axes as the original panel E: thick L-shaped spines, no ticks."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.1)
        ax.spines[side].set_color(INK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.minorticks_off()
    ax.set_xlabel(xlabel, fontsize=9.5, labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, labelpad=3)


def arrow(fig, x0: float, x1: float, y: float, label: str) -> None:
    fig.add_artist(
        mpl.patches.FancyArrowPatch(
            (x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=8, linewidth=1.3, color=INK,
        )
    )
    fig.text((x0 + x1) / 2, y + 0.05, label, ha="center", va="bottom",
             fontsize=6.3, style="italic", color=INK, linespacing=1.1)


def draw_panel() -> None:
    plt.rcParams.update(
        {"font.family": "sans-serif", "font.sans-serif": ["Arial"], "pdf.fonttype": 42,
         # equation in Arial too
         "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
         "mathtext.bf": "Arial:bold"}
    )
    t_raw, y_raw = load_raw()
    t, y = median_per_count(t_raw, y_raw)
    d_eff = (
        effective_dimensions()
        .filter((pl.col("site") == SITE) & (pl.col("energy") == float(KT)))["dim"]
        .item()
    )

    # the fit: d_eff exponentials (mix of the floor/ceil fits, as Fig. 2A);
    # its components are drawn from the floor(d_eff)-exponential fit
    n = int(np.floor(d_eff))
    grid = np.arange(1, t.max() + 1)
    total = effective_fit(t, y, d_eff)
    q = fit_n_exp(t, y, n)
    amps, rates = np.exp(q[:n]), np.exp(q[n:])
    order = np.argsort(rates)[::-1]  # fastest first

    width, height = E_RECT.width / 72, E_RECT.height / 72
    fig = plt.figure(figsize=(width, height))
    bottom, top = 0.17, 0.95
    lefts, w = (0.075, 0.405, 0.735), 0.245
    xlim, ylim = (0.8, t_raw.max() * 1.4), (0.9, y.max() * 2.5)
    axes = [fig.add_axes([l, bottom, w, top - bottom]) for l in lefts]

    # a few representative points, as the original sketch: real values (except the
    # schematic spread of the tail), picked
    # evenly along log(duration) so the shape of the curve stays true
    head = y > 45  # the smooth part of the curve; low counts form the scattered tail
    curve_t, curve_y = representative(t[head], y[head], n=10)
    # the scattered tail of the raw data: several durations share the same low
    # counts; the median step collapses each count to one point
    tail_t, tail_y, med_t, med_y = [], [], [], []
    for c in (16, 6, 2):
        median = np.median(t_raw[y_raw == c])
        # schematic: three events with this count, spread around their true
        # median so the fan (and its collapse to the middle point) is visible
        tail_t += list(median * np.array([0.6, 1.0, 1.65]))
        tail_y += [c] * 3
        med_t.append(median)
        med_y.append(c)
    marker = dict(color=RED, edgecolors=INK, linewidths=0.5, zorder=2)

    # 1) raw distribution
    ax = axes[0]
    ax.scatter(np.r_[curve_t, tail_t], np.r_[curve_y, tail_y], s=16, **marker)
    style_axes(ax, ylabel="Occurrence")

    # 2) median duration per distinct count
    ax = axes[1]
    ax.scatter(np.r_[curve_t, med_t], np.r_[curve_y, med_y], s=16, **marker)
    style_axes(ax)

    # 3) fit with a sum of exponentials, components underneath
    ax = axes[2]
    for c, i in zip(COMPONENT_COLORS, order):
        comp = amps[i] * np.exp(-rates[i] * grid)
        ok = comp >= ylim[0]
        ax.plot(grid[ok], comp[ok], color=c, linewidth=1.0, linestyle=(0, (3, 1.5)), zorder=1)
    ax.scatter(np.r_[curve_t, med_t], np.r_[curve_y, med_y], s=16, **marker)
    ok = total >= ylim[0]
    ax.plot(grid[ok], total[ok], color=INK, linewidth=1.5, linestyle=(0, (4, 1.6)), zorder=3)
    style_axes(ax)
    # equation in the empty lower-left corner (the decay runs top-left to bottom-right)
    ax.text(0.05, 0.27, r"$f(t)=\sum_{i=1}^{d}\,a_i\,e^{-k_i t}$", transform=ax.transAxes,
            ha="left", va="center", fontsize=7.5, color=INK)
    ax.text(0.05, 0.075, rf"$d_{{\mathrm{{eff}}}} = {np.floor(d_eff * 100) / 100:.2f}$",
            transform=ax.transAxes, ha="left", va="center", fontsize=7.5, color=INK)

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    y_arrow = bottom + 0.45 * (top - bottom)
    arrow(fig, lefts[0] + w + 0.015, lefts[1] - 0.015, y_arrow, "Median\ncalculation")
    arrow(fig, lefts[1] + w + 0.015, lefts[2] - 0.015, y_arrow, "Multi-exp.\nfitting")

    fig.savefig(PANEL, transparent=True)  # exact page size, no tight bbox
    plt.close(fig)


def splice() -> None:
    """Original page with panel E replaced; panels A-D untouched."""
    doc = pymupdf.open(ORIGINAL)
    page = doc[0]
    for rect in (E_RECT, TAG_GAP):
        page.draw_rect(rect, color=None, fill=(1, 1, 1), overlay=True)
    page.show_pdf_page(E_RECT, pymupdf.open(PANEL), 0)
    doc.save(OUTPUT, garbage=3, deflate=True)
    doc[0].get_pixmap(dpi=300).save(OUTPUT.replace(".pdf", ".png"))


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    draw_panel()
    splice()
    print(OUTPUT)


if __name__ == "__main__":
    main()
