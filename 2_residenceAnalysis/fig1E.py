"""Figure 1E, redrawn from data and spliced into the Illustrator-made Figure 1.

Panel E shows how a residence-time distribution becomes an effective dimension:
distribution (median duration per distinct count) -> fit with a sum of n
exponentials -> the number of exponentials needed, averaged over many fits, is
the (non-integer) effective dimension d_eff = <n>.

The distribution points are a few representative values from NS sites, 3.5 kT,
60 µM; the third step is schematic (no specific case).

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
from fig2 import fit_n_exp  # noqa: E402  (same fitting as Fig. 2A)
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
BAR = "#2e3192"  # Fig. 1 navy


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
         # equations in Arial too
         "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
         "mathtext.bf": "Arial:bold"}
    )
    t_raw, y_raw = load_raw()
    t, y = median_per_count(t_raw, y_raw)

    # one fit with n exponentials (n = 3 here), drawn with its components
    n = 3
    grid = np.arange(1, t.max() + 1)
    q = fit_n_exp(t, y, n)
    amps, rates = np.exp(q[:n]), np.exp(q[n:])
    order = np.argsort(rates)[::-1]  # fastest first
    total = (amps[:, None] * np.exp(-np.outer(rates, grid))).sum(axis=0)

    width, height = E_RECT.width / 72, E_RECT.height / 72
    fig = plt.figure(figsize=(width, height))
    bottom, top = 0.17, 0.95
    lefts, w = (0.075, 0.395, 0.77), 0.225
    xlim, ylim = (0.8, t.max() * 1.6), (0.9, y.max() * 2.5)
    axes = [fig.add_axes([l, bottom, w, top - bottom]) for l in lefts]

    # a few representative points, as the original sketch: real values (median
    # duration per count), picked evenly along log(duration) so the shape stays true
    head = y > 45
    curve_t, curve_y = representative(t[head], y[head], n=10)
    tail = [np.median(t_raw[y_raw == c]) for c in (16, 6, 2)]
    pts_t, pts_y = np.r_[curve_t, tail], np.r_[curve_y, [16, 6, 2]]
    marker = dict(s=16, color=RED, edgecolors=INK, linewidths=0.5, zorder=2)

    # 1) distribution (median duration per distinct count)
    ax = axes[0]
    ax.scatter(pts_t, pts_y, **marker)
    style_axes(ax, ylabel="Occurrence")

    # 2) fit with a sum of n exponentials, components underneath
    ax = axes[1]
    for c, i in zip(COMPONENT_COLORS, order):
        comp = amps[i] * np.exp(-rates[i] * grid)
        ok = comp >= ylim[0]
        ax.plot(grid[ok], comp[ok], color=c, linewidth=1.0, linestyle=(0, (3, 1.5)), zorder=1)
    ax.scatter(pts_t, pts_y, **marker)
    ok = total >= ylim[0]
    ax.plot(grid[ok], total[ok], color=INK, linewidth=1.5, linestyle=(0, (4, 1.6)), zorder=3)
    style_axes(ax)
    # equation in the empty lower-left corner (the decay runs top-left to bottom-right)
    ax.text(0.05, 0.2, r"$f(t)=\sum_{i=1}^{n}\,a_i\,e^{-k_i t}$", transform=ax.transAxes,
            ha="left", va="center", fontsize=7.5, color=INK)

    for ax in axes[:2]:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    # 3) effective dimension: the number of exponentials n needed, averaged over
    # many fits. Schematic (illustrative fractions, no specific case)
    ax = axes[2]
    ns = np.arange(1, 6)
    frac = np.array([0.03, 0.14, 0.48, 0.27, 0.08])  # skewed: <n> falls between integers
    mean = (ns * frac).sum()
    ax.bar(ns, frac, width=0.62, color=BAR, edgecolor=INK, linewidth=0.5, zorder=2)
    ax.axvline(mean, color=RED, linewidth=1.4, linestyle=(0, (3, 1.5)), zorder=3)
    ax.text(mean + 0.22, frac.max() * 1.12, r"$d_{\mathrm{eff}} = \langle n \rangle$",
            ha="left", va="center", fontsize=7.5, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.1)
        ax.spines[side].set_color(INK)
    ax.set_xticks(ns)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=2.5, width=0.8, labelsize=7, pad=1.5)
    ax.set_xlim(0.4, 5.6)
    ax.set_ylim(0, frac.max() * 1.3)
    ax.set_xlabel("Exponentials, $n$", fontsize=9.5, labelpad=1)
    ax.set_ylabel("Fits", fontsize=9.5, labelpad=2)

    y_arrow = bottom + 0.45 * (top - bottom)
    arrow(fig, lefts[0] + w + 0.015, lefts[1] - 0.015, y_arrow, "Multi-exp.\nfitting")
    arrow(fig, lefts[1] + w + 0.012, lefts[2] - 0.05, y_arrow, "Averaging\nover fits")

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
