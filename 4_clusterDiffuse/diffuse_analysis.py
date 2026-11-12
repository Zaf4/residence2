import numpy as np
import pandas as pd
import cluster_analysis_extra as clust
import os
import platform
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy.spatial import ConvexHull
from rich.progress import track


def findResidences(arr: np.ndarray) -> np.ndarray:
    """
    Finds residence time of particles.

    Parameters
    ----------
    arr : np.ndarray
        State array (full).

    Returns
    -------
    rt : np.ndarray
        Residence Times.

    """

    # allocatin an array
    rt = np.zeros([len(arr)])
    for i, row in enumerate(arr):
        a = np.where(row == 0)[0]
        if len(a) > 0:
            rt[i] = a[0]
        else:
            rt[i] = len(arr[0])

    return rt


def count_bounds(arr: np.ndarray) -> np.ndarray:
    """
    Counts the remaining bound particles.

    Parameters
    ----------
    arr : np.ndarray
        State array.

    Returns
    -------
    decay : np.ndarray
        Array of number of remaining bound particles.

    """

    rt = findResidences(arr)
    rt = np.sort(rt).astype(int)
    bound = len(arr)
    decay = np.ones([arr.shape[1]], dtype=np.float32) * bound

    # finding decay from residences..
    for i, r in enumerate(rt):
        bound -= 1
        decay[r:] = bound

    decay[decay == 0] = np.nan

    return decay


def removeDNA(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.type > 2]


def addAtomIndex(df: pd.DataFrame) -> pd.DataFrame:
    if "atomID" not in df:
        tfatom = len(df)
        df["atomID"] = np.arange(tfatom) + 1  # assingning atomID
    return df


def addTFindex(df: pd.DataFrame) -> pd.DataFrame:
    if "tfID" not in df:
        # number of tf = number of tf atoms / 3
        tfmol = int(len(df) / 3)
        # tf indexing
        tfIDs = np.repeat(np.arange(tfmol) + 1, 3)
        df["tfID"] = tfIDs
    return df


def findDecayRate_old(decay: np.ndarray) -> float:
    """
    Using single exponential decay function finds meanlifetime of group of tfs
    or cluster of tfs.

    Parameters
    ----------
    decay : np.ndarray
        Decay array N0-->0.

    Returns
    -------
    float
        tau or meanlifetime or 1/K where K is unbinding rate.

    """

    def exp_decay(x, a, b):
        return a * np.exp(-x * b)

    decay = decay[decay > decay[0] / 10]  # below 10 percent is not counted

    timestep_full = np.arange(len(decay)) + 1
    timestep, decay = timestep_full[decay > 0], decay[decay > 0]

    popt, pcov = curve_fit(exp_decay, timestep, decay, maxfev=2000000)
    coeff, K = popt
    tau = 1 / K

    return tau


def findDecayRate(decay: np.ndarray, dt: float = 1.0) -> float:
    """
    Mean lifetime (tau) of a group/cluster of tfs as the area under the
    survival curve, making no assumption about the shape of the decay.

    The mean lifetime of a non-negative lifetime T equals the integral of its
    survival function S(t) = N(t)/N0:

        tau = E[T] = integral_0^inf S(t) dt = (1/N0) * integral_0^inf N(t) dt

    This is exact for any decay shape; for a pure single exponential
    N(t) = N0*exp(-K t) it reduces to tau = 1/K, matching findDecayRate_old.

    The full curve (down to 0) is integrated -- unlike the old function, the
    slow tail is NOT truncated, since dropping it would discard real area and
    bias tau low.

    Parameters
    ----------
    decay : np.ndarray
        Survival curve N(t): number of bound tfs vs time, N0-->0.
    dt : float
        Time per index (timestep). tau is returned in the same time units.

    Returns
    -------
    float
        tau or meanlifetime. Equals 1/K for a pure exponential.

    """

    # count_bounds marks fully-unbound (0 remaining) as NaN; for the integral
    # that is genuinely 0 bound particles, not missing data.
    decay = np.nan_to_num(decay, nan=0.0)

    N0 = decay[0]
    if N0 <= 0:
        return 0.0

    # area under N(t), normalized by N0 -> integral of S(t)
    tau = np.trapz(decay, dx=dt) / N0

    return tau


def findDecayRate_tail_corrected(
    decay: np.ndarray, dt: float = 1.0, tail_frac: float = 0.5
) -> float:
    """
    Mean lifetime (tau) as the area under the survival curve, plus an analytic
    correction for the area beyond the observation window.

    When the window ends before the cluster has fully unbound (right-censored
    decay), N(t) never reaches 0 and the bare integral underestimates tau. We
    add the remaining area of an exponential tail fitted to the last portion of
    the curve:

        tau = (1/N0) * [ integral_0^tend N(t) dt + N(tend)/K ]

    where K is the local unbinding rate from fitting a*exp(-K t) to the final
    `tail_frac` of the survival curve. The remainder N(tend)/K is the integral
    of N(tend)*exp(-K (t - tend)) from tend to infinity.

    If the curve already reaches ~0 the correction term is negligible and this
    matches findDecayRate.

    Parameters
    ----------
    decay : np.ndarray
        Survival curve N(t): number of bound tfs vs time, N0-->0.
    dt : float
        Time per index (timestep). tau is returned in the same time units.
    tail_frac : float
        Fraction of the curve (from the end) used to fit the tail rate K.

    Returns
    -------
    float
        tau or meanlifetime, tail-corrected.

    """

    # count_bounds marks fully-unbound (0 remaining) as NaN; treat as 0 bound.
    decay = np.nan_to_num(decay, nan=0.0)

    N0 = decay[0]
    if N0 <= 0:
        return 0.0

    t = np.arange(len(decay)) * dt
    measured_area = np.trapz(decay, dx=dt)

    N_end = decay[-1]

    # if effectively fully decayed, no tail correction needed
    if N_end <= N0 / 1000:
        return measured_area / N0

    def exp_decay(x, a, b):
        return a * np.exp(-x * b)

    # fit a*exp(-K t) to the last tail_frac of the curve to get the local rate K
    start = int(len(decay) * (1 - tail_frac))
    t_tail, decay_tail = t[start:], decay[start:]

    try:
        popt, _ = curve_fit(
            exp_decay,
            t_tail,
            decay_tail,
            p0=(N_end, 1.0 / (t[-1] + dt)),
            maxfev=2000000,
        )
        K = popt[1]
    except (RuntimeError, ValueError):
        K = 0.0

    # remaining area of the exponential tail beyond the window: N_end / K
    tail_area = N_end / K if K > 0 else 0.0

    tau = (measured_area + tail_area) / N0

    return tau


def addClusterLifetimes(
    df: pd.DataFrame,
    states: np.ndarray,
    timestep: int = 1000,
    duration: int = 2000,
    primary: str = "tail",
) -> pd.DataFrame:
    """
    Compute per-cluster mean lifetimes with all three methods in a single pass
    and attach them as columns:

        tau_old  : single-exponential fit          (findDecayRate_old)
        tau_int  : area under survival curve        (findDecayRate)
        tau_tail : area + censored-tail correction  (findDecayRate_tail_corrected)

    All three are derived from the SAME decay curve, so the CSV can be compared
    method-vs-method later without recomputing from the raw states. The `tau`
    column is an alias to the chosen `primary` method ("old" | "int" | "tail")
    so existing plotting code keeps working unchanged.
    """

    n = len(df)
    taus_old = np.zeros(n, dtype=np.float32)
    taus_int = np.zeros(n, dtype=np.float32)
    taus_tail = np.zeros(n, dtype=np.float32)

    # for each cluster --also nonclusters-- add decay rates for that cluster
    for i in range(int(df.clusterID.max()) + 1):
        # isolating a cluster
        single_cluster = df[df.clusterID == i]
        tfIDs = np.unique(single_cluster.tfID)
        atomIDs = np.unique(single_cluster.atomID)

        # for only tfs in the given --cluster-- and --timestep--
        partial = states[tfIDs - 1, timestep : timestep + duration]

        # one decay curve, three lifetime estimates
        decay = count_bounds(partial)
        try:
            t_old = findDecayRate_old(decay)
        except (RuntimeError, ValueError):
            # single-exp fit can fail to converge; flag instead of crashing
            t_old = np.nan
        t_int = findDecayRate(decay)
        t_tail = findDecayRate_tail_corrected(decay)

        # assigning taus to relevant places
        taus_old[atomIDs - 1] = t_old
        taus_int[atomIDs - 1] = t_int
        taus_tail[atomIDs - 1] = t_tail

    # add per-method columns
    df["tau_old"] = taus_old
    df["tau_int"] = taus_int
    df["tau_tail"] = taus_tail

    # `tau` alias used by the existing plotting / regression code
    primary_map = {"old": taus_old, "int": taus_int, "tail": taus_tail}
    if primary not in primary_map:
        raise ValueError(f"primary must be one of {list(primary_map)}, got {primary!r}")
    df["tau"] = primary_map[primary]

    return df


def addResidenceTimes(
    df: pd.DataFrame, states: np.ndarray, timestep: int = 1000
) -> pd.DataFrame:
    # find residences and assing it to their respective tfs
    rt_tf = findResidences(states[:, timestep:])
    rt_atoms = np.repeat(rt_tf, 3)

    df["residence"] = rt_atoms

    return df


def addIndexes(df: pd.DataFrame):

    df = removeDNA(df)  # removing DNA
    df = addAtomIndex(df)  # atom indexing
    df = addTFindex(df)  # TF indexing

    return df


def addClusterSizes(df: pd.DataFrame) -> pd.DataFrame:
    # initing taus array
    sizes = np.ones(len(df), dtype=np.int32)

    # for each cluster finds its size and assing it to atoms of it
    for i in range(1, int(df.clusterID.max()) + 1):
        # isolating a cluster
        single_cluster = df[df.clusterID == i]
        size = int(len(single_cluster) / 3)
        atomIDs = single_cluster.atomID
        sizes[atomIDs - 1] = size

    df["clusterSize"] = sizes

    return df


def addTimestep(df: pd.DataFrame, timestep: int) -> pd.DataFrame:
    ts_arr = np.zeros(len(df), dtype=np.int32)
    ts_arr[:] = timestep
    df["timestep"] = ts_arr
    return df


def addClusterShape(df: pd.DataFrame) -> pd.DataFrame:

    num_clust = df.clusterID.max()

    conformations = np.repeat(["None" + " " * 12], len(df))
    for i in np.arange(num_clust) + 1:
        partial = df[df.clusterID == i]
        atomIDs = partial.atomID
        coor = np.array(partial[["x", "y", "z"]])
        shape = clust.find_cluster_shape(coor)
        conformations[atomIDs - 1] = shape

    conformations = [x.strip() for x in conformations]
    df["conformation"] = conformations

    return df


def addKT(df: pd.DataFrame, kT: float) -> pd.DataFrame:
    kts = np.zeros(len(df))
    kts[:] = kT
    df["kT"] = kts

    return df


def addSurface(df: pd.DataFrame) -> pd.DataFrame:
    num_clust = df.clusterID.max()

    # initing surface array
    surface = np.zeros(len(df))  # 0 for the free atoms
    # timesteps =np.sort(df.timestep.unique())

    for i in np.arange(num_clust) + 1:
        partial = df[df.clusterID == i]
        atomIDs = np.array(
            partial.atomID
        )  # storing indexes before removing binding domains
        # marking core atoms
        surface[atomIDs - 1] = 2  # 2 for the core atoms

        # removing non type-5 (hinge domain) particles
        partial = partial[partial.type == 5]
        coor = np.array(partial[["x", "y", "z"]])

        # resetting atomIDs after taking subset of df
        atomIDs = np.array(partial.atomID)

        # using convexHull algorithm finding surface atoms
        hull = ConvexHull(coor, qhull_options="QJ Tv 1e-12", incremental=True)
        surf_ids = atomIDs[np.array(hull.vertices)] - 1

        # marking surface tfs
        ##hinge domain
        surface[surf_ids] = 1  # 1 for surface
        ##binding domains
        surface[surf_ids - 1] = 1
        surface[surf_ids + 1] = 1

    df["kind"] = surface

    return df


def load_states(statesfile: str | os.PathLike) -> np.ndarray:
    """
    Load and decode a raw full.npy state file into a per-TF bound/unbound array.

    The raw file has two rows per TF (left and right arm). A TF counts as bound
    only when BOTH arms are bound, so the two rows are summed (==2 -> bound) and
    collapsed to one row per TF.

    Parameters
    ----------
    statesfile : os.PathLike
        Path to a raw full.npy file (rows = 2*nTF, columns = time).

    Returns
    -------
    np.ndarray
        Boolean array, rows = nTF, columns = time. states[tfID-1] is one TF.
    """

    states = np.load(statesfile)
    stateU = states[1::2].astype(int) + states[0::2].astype(int)
    stateU[stateU == 1] = 0
    stateU[stateU == 2] = 1
    return stateU.astype(bool)


def recompute_taus(
    df: pd.DataFrame,
    statesfile: str | os.PathLike,
    duration: int = 300,
    primary: str = "tail",
) -> pd.DataFrame:
    """
    Recompute the tau_old / tau_int / tau_tail columns for an already-built
    dataframe, using the raw states file but reusing the existing cluster
    assignments (clusterID / tfID / atomID / timestep) in df.

    This avoids rerunning clustering and surface analysis: only the lifetime
    estimates are redone, per timestep, with the current findDecayRate* code.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe from a saved CSV (must contain clusterID, tfID, atomID,
        timestep).
    statesfile : os.PathLike
        Path to the matching raw full.npy file.
    duration : int
        Decay window length (must match the value used originally; default 300).
    primary : str
        Which method the `tau` alias points to ("old" | "int" | "tail").

    Returns
    -------
    pd.DataFrame
        df with tau_old / tau_int / tau_tail / tau recomputed.
    """

    states = load_states(statesfile)
    parts = [
        addClusterLifetimes(
            group.copy(),
            states,
            timestep=int(ts),
            duration=duration,
            primary=primary,
        )
        for ts, group in df.groupby("timestep")
    ]
    return pd.concat(parts)


def generateDF(
    dumpfile: os.PathLike = "./data/targetDUMP.npy",
    statesfile: os.PathLike = "./data/full.npy",
    timestep=2100,
    kT: float = 1.23,
) -> pd.DataFrame:
    """generates a multitime-modified dataframe

    Parameters
    ----------
    dumpfile : os.PathLike, optional
        path/to/dump/file, by default './data/targetDUMP.npy'
    statesfile : os.PathLike, optional
        path/to/states/file, by default './data/full.npy'
    timestep : int, optional
        timestep label for dataframe, by default 2100
    kT : float, optional
        affinity label, by default 1.23

    Returns
    -------
    pd.DataFrame
        _description_
    """

    # load the dump file 3D, 0=timestep,1=atoms,2=attributes (type,x,y,z)
    dump = np.load(dumpfile)
    df = clust.cluster_single_frame(dump[timestep])

    # load residence states rows=proteins, columns=time
    states = load_states(statesfile)

    # adding indexes
    df = addIndexes(df)
    # adding mean life times (clusters)
    df = addClusterLifetimes(df, states, timestep=timestep, duration=300)
    # adding residence times (all atoms)
    df = addResidenceTimes(df, states, timestep=timestep)
    # adding cluster sizes
    df = addClusterSizes(df)
    # adding timesteps
    df = addTimestep(df, timestep=timestep)
    # add cluster shape
    df = addClusterShape(df)
    # add kTs (energies)
    df = addKT(df, kT)
    # add kind (surface,core,free)
    df = addSurface(df)

    return df


def multiTimeDF(timesteps: np.ndarray, **kwargs) -> pd.DataFrame:

    # initing empty dataframe
    df = pd.DataFrame()
    for timestep in track(timesteps):
        ts = generateDF(timestep=timestep, **kwargs)
        df = pd.concat([df, ts])

    return df


def makePlot(df: pd.DataFrame, *args, **kwargs) -> plt.Axes:
    """_summary_

    Parameters
    ----------
    df : pd.DataFrame
        data

    Returns
    -------
    plt.Axes
        plot ax
    """

    # font = 1
    sns.set_theme(
        style="ticks",
        rc={
            "font.weight": "light",
            "font.size": 14,
            "font.family": "Arial",
            "ytick.minor.size": "0",
            "ytick.major.size": "10",
            "xtick.major.size": "10",
        },
    )

    df = df[df.clusterID > 0]
    max_time = df.timestep.min()
    N = len(np.unique(df.timestep))
    fig, ax = plt.subplots(
        1, 2, figsize=(16, 4.5), gridspec_kw={"width_ratios": [2.5, 1]}
    )
    # showing the system -------------SCATTER-----------------------------------
    sns.scatterplot(
        df[df.timestep == max_time],
        x="x",
        y="y",
        hue="tau",
        palette="Purples",
        edgecolor="k",
        ax=ax[0],
    )
    ax[0].annotate(
        xy=(0.05, 0.1),
        text=f"t = {max_time} a.u",
        fontweight="bold",
        xycoords="axes fraction",
    )
    ax[0].set_xlim([-90, 90])
    ax[0].set_ylim([-30, 30])
    # adding color bar
    norm = plt.Normalize(df.tau.min(), df.tau.max())
    purples = plt.cm.ScalarMappable(cmap="Purples", norm=norm)
    purples.set_array([])
    ax[0].get_legend().remove()
    ax[0].figure.colorbar(
        purples, ax=ax[0], location="right", shrink=1, label="Mean Lifetime"
    )
    ax[0].figure.tick_params(labelsize=30)

    # regression  -----------------REGRESSION-----------------------------------
    sns.regplot(df, x="clusterSize", y="tau", ax=ax[1], color="red")
    # finding correlation coeff
    r, p = stats.pearsonr(df["clusterSize"], y=df["tau"])
    ax[1].annotate(
        xy=(0.05, 0.90),
        text=f"r = {r:.2f}",
        fontweight="bold",
        xycoords="axes fraction",
    )
    ax[1].annotate(
        xy=(0.85, 0.05), text=f"N = {N}", fontweight="bold", xycoords="axes fraction"
    )
    ax[1].set_ylabel("Mean Lifetime")
    ax[1].set_xlabel("Cluster Size")
    fig.tight_layout()

    return ax


def simplify(df: pd.DataFrame) -> pd.DataFrame:
    """removes not needed columns and duplicate clusters

    Parameters
    ----------
    df : pd.DataFrame
        dataframe

    Returns
    -------
    pd.DataFrame
        modified dataframe
    """

    keep = [
        "clusterID",
        "residence",
        "clusterSize",
        "timestep",
        "conformation",
        "kT",
        "tau",
        "tau_old",
        "tau_int",
        "tau_tail",
    ]
    # keep tau_* only when present so older CSVs (single tau column) still work
    df = df[[c for c in keep if c in df.columns]]
    df = df.drop_duplicates(["clusterID", "timestep"])
    df = df[df.clusterID > 0]
    df = df[df.residence > 0]
    return df


def makePlotMulti_tau(dfs: list[pd.DataFrame], *args, **kwargs) -> plt.Axes:
    """
    takes multiple DataFrames and make multiple rows
    of figures showing clusters and lifetimes

    Parameters
    ----------
    dfs : list
        list of dataframes

    Returns
    -------
    plt.Axes
        figure
    """

    # loading capsid df
    capsid = pd.read_csv("./data/capsid2d.csv", index_col=None)

    # clear graph settings
    sns.set_theme(
        style="ticks",
        rc={
            "font.weight": "light",
            "font.size": 21,
            "font.family": "sans-serif",
            "ytick.minor.size": "0",
            "ytick.major.size": "6",
            "xtick.major.size": "6",
        },
    )

    num_rows = len(dfs)
    fig, ax = plt.subplots(
        num_rows, 2, figsize=(16, 4 * num_rows), gridspec_kw={"width_ratios": [2.25, 1]}
    )
    kts = [2.80, 3.00, 3.50, 4.00]

    # one colour scale for the whole of panel A; lifetimes span ~2 decades
    # across energies, so a log scale keeps the weak-binding rows readable
    all_tau = pd.concat([d.loc[d.clusterID > 0, "tau"] for d in dfs])
    norm = mpl.colors.LogNorm(all_tau[all_tau > 0].min(), all_tau.max())

    for row, df in enumerate(dfs):
        df = df[df.clusterID > 0]
        time = np.random.choice(df.timestep.unique())
        N = len(np.unique(df.tau))

        kt = kts[row]
        # plotting the -----------------CAPSID------------------
        sns.scatterplot(
            data=capsid,
            color="#900C3F",
            ax=ax[row, 0],
            x="x",
            y="y",
            edgecolor="#900C3F",
            size=20,
        )

        # plotting the system -------------SCATTER-----------------------------------
        sns.scatterplot(
            data=df[df.timestep == time],
            x="x",
            y="y",
            hue="tau",
            palette="Purples",
            hue_norm=norm,
            edgecolor="k",
            ax=ax[row, 0],
            linewidth=0.3,
        )
        ax[row, 0].annotate(
            xy=(0.12, 0.06),
            fontsize=16,
            text=f"t = {time}",
            fontweight="light",
            style="italic",
            xycoords="axes fraction",
        )
        # ax[row,0].annotate(xy=(0.12,0.90), fontsize =20,
        #          text=f'{kt:.1f}kT',
        #         fontweight='bold', style = 'italic',
        #        xycoords='axes fraction')
        # ax[row,0].set_title(f'{kt:.1f}kT',fontweight='light', fontsize=24)
        ax[row, 0].set_ylabel("")
        ax[row, 0].set_xlabel("")
        ax[row, 0].set_xlim([-92, 92])
        ax[row, 0].set_ylim([-31, 31])
        ax[row, 0].set_xticks([])
        ax[row, 0].set_yticks([])
        ax[row, 0].get_legend().remove()
        sns.despine(ax=ax[row, 0], left=True, bottom=True)
        sns.despine(ax=ax[row, 1])
        # regression  -----------------REGRESSION-----------------------------------
        df_simple = simplify(df)

        # overlay the three lifetime estimates; fall back to the single `tau`
        # column for older CSVs that don't carry the per-method columns.
        methods = [
            ("tau_old", "single-exp", "red"),
            ("tau_int", "integral", "tab:blue"),
            ("tau_tail", "integral+tail", "green"),
        ]
        methods = [m for m in methods if m[0] in df_simple.columns]
        if not methods:
            methods = [("tau", "tau", "red")]

        for j, (col, label, color) in enumerate(methods):
            # drop failed single-exp fits (inf / non-positive) before plotting
            m = df_simple[["clusterSize", col]].replace([np.inf, -np.inf], np.nan)
            m = m[m[col] > 0].dropna()
            if len(m) < 2:
                continue

            sns.regplot(
                data=m,
                x="clusterSize",
                y=col,
                ax=ax[row, 1],
                color=color,
                label=label,
                scatter_kws={"s": 12, "alpha": 0.4},
                line_kws={"linewidth": 2},
            )

            # pearson r + slope per method, stacked top-left
            r, _ = stats.pearsonr(m["clusterSize"], m[col])
            slope = np.polyfit(m["clusterSize"], m[col], 1)[0]
            ax[row, 1].annotate(
                xy=(0.05, 0.90 - 0.09 * j),
                text=f"{label}: r={r:.2f}, m={slope:.1f}",
                color=color,
                fontsize=12,
                fontweight="light",
                style="italic",
                xycoords="axes fraction",
            )

        # showing number of timepoints
        ax[row, 1].annotate(
            xy=(0.75, 0.05),
            text=f"N = {N}",
            fontweight="light",
            style="italic",
            xycoords="axes fraction",
        )

        ax[row, 1].legend(fontsize=10, loc="center right", frameon=False)
        ax[row, 1].tick_params(axis="both", labelsize=14)
        ax[row, 1].set_ylabel("Mean Lifetime (a.u.)", fontsize=18)
        ax[row, 1].set_xlabel("Cluster Size (#TF)", fontsize=18)

    fig.tight_layout(rect=(0.07, 0, 1, 0.97))  # room for the colour bar and tags

    # single colour bar spanning all rows of panel A, left of the snapshots
    top, bottom = ax[0, 0].get_position(), ax[-1, 0].get_position()
    cax = fig.add_axes([0.03, bottom.y0 + 0.02, 0.012, top.y1 - bottom.y0 - 0.04])
    purples = plt.cm.ScalarMappable(cmap="Purples", norm=norm)
    cbar = fig.colorbar(purples, cax=cax, extend="both")
    cbar.set_ticks([t for t in (5, 10, 20, 50, 100, 200, 500) if norm.vmin <= t <= norm.vmax])
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    cbar.ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Mean Lifetime (a.u.)", fontsize=18)

    # column tags, top-left of each column (A over the colour bar, B over the y label)
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    y_tag = top.y1 + 0.005
    x_a = to_fig.transform(cbar.ax.get_tightbbox().p0)[0]
    x_b = to_fig.transform(ax[0, 1].get_tightbbox().p0)[0]
    for x, tag in ((x_a, "A"), (x_b, "B")):
        fig.text(x, y_tag, tag, fontsize=40, fontweight="bold", ha="left", va="bottom")

    return fig


if __name__ == "__main__":
    """
    #___________trial with small data on PC________________
    timesteps = np.random.random_integers(1500,4000,size=2)
    df = multiTimeDF(timesteps,
                     dumpfile='./data/targetDUMP.npy',
                     statesfile='./data/full.npy',
                     kT=3.50)
    
    df2 = multiTimeDF(timesteps,
                     dumpfile='./data/targetDUMP.npy',
                     statesfile='./data/full.npy',
                     kT=2.80)   
    
    df3 = multiTimeDF(timesteps,
                     dumpfile='./data/targetDUMP.npy',
                     statesfile='./data/full.npy',
                     kT=3.10)   
    
    
    
    ax = makePlotMulti([df,df2,df3])
    plt.savefig('./graphs/multiLove.png', dpi=400)
    """

    if "celestia" in platform.node():  # run this part only if pc name includes celestia
        # ____________this part was run on the HPC_______________
        timesteps = np.random.random_integers(5000, 22000, size=12)

        # for server
        flocs = "/home/gottar/5x10t/kt/60"
        saveloc = "/home/gottar/5x10t/graphs"
        kts = ["2.80", "3.00", "3.50", "4.00"]

        # file list
        dumpfiles = [flocs.replace("kt", kt) + "/dump.npy" for kt in kts]
        statesfiles = [flocs.replace("kt", kt) + "/full.npy" for kt in kts]

        # generating dfs
        df280 = multiTimeDF(
            timesteps,
            dumpfile=dumpfiles[0],
            statesfile=statesfiles[0],
            kT=float(kts[0]),
        )
        df300 = multiTimeDF(
            timesteps,
            dumpfile=dumpfiles[1],
            statesfile=statesfiles[1],
            kT=float(kts[1]),
        )
        df350 = multiTimeDF(
            timesteps,
            dumpfile=dumpfiles[2],
            statesfile=statesfiles[2],
            kT=float(kts[2]),
        )
        df400 = multiTimeDF(
            timesteps,
            dumpfile=dumpfiles[3],
            statesfile=statesfiles[3],
            kT=float(kts[3]),
        )
        # save dfs for further uses
        df280.to_csv(f"{saveloc}/280.csv", index=False)
        df300.to_csv(f"{saveloc}/300.csv", index=False)
        df350.to_csv(f"{saveloc}/350.csv", index=False)
        df400.to_csv(f"{saveloc}/400.csv", index=False)

        # #plotting and saving the plot
        # ax = makePlotMulti([df280,df300,df350,df400])
        # plt.savefig(f'{saveloc}/multi.png', dpi=400)

    else:
        # _______this part is done on PC to data obtained from Cluster_________
        # changing working directory to current directory name
        os.chdir(os.path.dirname(__file__))

        # kT label -> (csv file, matching raw full.npy). The 12timepoint CSVs
        # were produced from the ".../kt/60/full.npy" runs (the -60 variants).
        kts = ["2.80", "3.00", "3.50", "4.00"]
        csvs = ["280", "300", "350", "400"]

        dfs = []
        for kt, csv in zip(kts, csvs):
            df = pd.read_csv(f"./data/12timepoint/{csv}.csv", index_col=False)

            # recompute tau_old/tau_int/tau_tail from local full.npy when present;
            # otherwise fall back to whatever tau is already in the CSV.
            statesfile = f"./data/fulls/{kt}-60.full.npy"
            if os.path.exists(statesfile):
                df = recompute_taus(df, statesfile, duration=300, primary="tail")

            dfs.append(df)

        fig = makePlotMulti_tau(dfs)

        fig.savefig("../Figures/fig4.pdf", transparent=True, bbox_inches="tight")
        # fig.savefig('../Figures/fig4.png', dpi=300, transparent=True)
        # ax1 = makePlotMulti_surface(dfs)
        # plt.savefig('../SI_Figures/multi_surface.pdf',transparent=True)
