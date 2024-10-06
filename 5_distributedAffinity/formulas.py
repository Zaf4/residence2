import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc

def double_exp(x, a, b, c, d):
    return a * np.exp(-x * b) + (d) * np.exp(-x * c)


def erfc_exp(t, A, tau, beta):
    return A * np.exp(-(t / tau) ** beta)


def erfc_exp2(t, a, koff, ks):
    return (a 
            * np.exp(( np.square(koff) * t) / ks) 
            * erfc(np.sqrt((np.square(koff) * t) / ks))
    )

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
