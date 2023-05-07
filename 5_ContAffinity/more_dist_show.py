import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import math
from scipy.optimize import curve_fit
from scipy import special

warnings.filterwarnings('ignore')

def deleteNaN(y:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    """
    delete NaN parts of the input array and time array opened for it,
    and returns time array and values array.

    """
    
    t = np.arange(len(y))+1
    val = np.array(y)
    
    t = t[~np.isnan(val)]
    val = val[~np.isnan(val)]
    

    return t,val


def double_exp(x,a,b,c,d):
    return a*np.exp(-x*b) + (d)*np.exp(-x*c)

def powerlaw(x,a,b):
	return a*x**(-b)

def complex_triple(x,a,b,c,d,e,f,cp,kp):
    return a*np.exp(-x*b) + c*np.exp(-x*d)+e*np.exp(-x*f)+cp*x**(-kp)

def complex_double(x,a,b,c,d,cp,kp):
    return a*np.exp(-x*b) + c*np.exp(-x*d)+cp*x**(-kp)

def st_exp(x,c,koff,ks):
        return c*np.exp(x*koff**2/ks)*special.erfc(np.sqrt(x*koff**2/ks))


def exp_decay(x,a,b):
    return a*np.exp(-x*b)

def tri_exp(x,a,b,c,d,e,f):
    return a*np.exp(-x*b) + c*np.exp(-x*d)+e*np.exp(-x*f)

def quad_exp(x,a,b,c,d,e,f,g,h):
    return a*np.exp(-x*b) + c*np.exp(-x*d)+e*np.exp(-x*f)+g*np.exp(-x*h)

def penta_exp(x,a,b,c,d,e,f,g,h,i,j):
    return a*np.exp(-x*b) + c*np.exp(-x*d)+e*np.exp(-x*f)+g*np.exp(-x*h)+i*np.exp(-x*j)

def value_fit(val:np.ndarray,eq:callable)->tuple[np.ndarray,np.ndarray,tuple]:
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
    
    t_range = np.arange(len(val))+1
    
    residual_t = np.zeros([len(val),2])
    
    t,val = deleteNaN(val)

    
    popt, pcov= curve_fit(eq, t, val, maxfev=20000000)
    residuals = (val- eq(t, *popt))
    ress_sumofsqr =np.sum(residuals**2)
    ss_res_norm = ress_sumofsqr/len(val)
    # ss_res_norm = ss_res/len(t)
    
    y_fit = eq(t_range, *popt)#full time length
    y_fit[y_fit<1] = np.nan#too small values to be removed
    y_fit[y_fit>np.max(val)*1.2] = np.nan#too big values removed
    
    return y_fit,ss_res_norm,popt

def arr_minimize(arr:np.ndarray,method:str='median')->np.ndarray:
    """
    Minimizes 1d array by removing repeats, according to the given method. 

    Parameters
    ----------
    arr : np.ndarray
        1d array to be minimized.
    method : str, optional
        'median' or 'average'. The default is 'median'.

    Returns
    -------
    arr1 : np.ndarray
        minimized array.

    """

    search = np.unique(arr) #arr of unique elements
    search = search[search>0] #remove nans
    
    arr1 = arr.copy()
    
    for s in search:
        positions, = np.where(arr==s)
        if method == 'median':
            mid = int(np.median(positions))
    
        elif method == 'average': 
            mid = int(np.average(positions))
        
        arr1[positions] = np.nan
        arr1[mid] = s #mid value is kept
        
    return arr1

def df_minimize(df:pd.DataFrame)->pd.DataFrame:
    """
    

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be minimized.

    Returns
    -------
    df : pd.DataFrame
        Minimized DataFrame.

    """
    for i in range(len(df.columns)):
        df.iloc[:,i] = arr_minimize(df.iloc[:,i]) #values minimized and returned

    return df



def scatterit_multi(df: pd.DataFrame, fits: pd.DataFrame,
                    i:int, j:int, axes,
                    palette: str, **kwargs) -> plt.Axes:
    """


    Parameters
    ----------
    df : pd.DataFrame
        Main data.
    fits : pd.DataFrame
        Equation fit of the main data.
    color: list
        Colors range.

    Returns
    -------
    ax : plot
        seaborn scatter graph.

    """
    
    
    ax = axes[i,j]

    # resetting indexes
    df.index = np.arange(len(df))+1
    fits.index = np.arange(len(df))+1

    # seting seaborn parameters
    font = {'family': 'Arial',
            'weight': 'light',
            'size': 14,
            }

    sns.set(style='ticks',
            palette=palette,
            rc={
                'font.weight': 'light',
                'font.family': 'sans-serif',
                'axes.spines.top': 'False',
                'axes.spines.right': 'False',
                'ytick.minor.size': '0',
                'xtick.minor.size': '0',
                'ytick.major.size': '10',
                'xtick.major.size': '10',
                'legend.frameon': False

                }
            )

    sns.set_palette(palette)
    for i, col in enumerate(df):
        # scatterplot
        sns.scatterplot(data=df[col], s=40,
                        # edgecolor=None,
                        alpha=0.4,
                        edgecolor='white', 
                        linewidth=0.01,
                        ax=ax,
                        **kwargs)
    for i, col in enumerate(fits):
        # lineplot
        sns.lineplot(data=fits[col],
                     ax=ax, linewidth=3,
                     # color='#1f1f1f',
                     linestyle='dashed', alpha=1, **kwargs)

    ax.tick_params(axis='both',labelsize=12)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.72, 3.7e3])
    ax.set_ylim([0.5, 1e6])
    
    
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    # ax.set_xlabel('Duration (a.u.)', fontdict=font)
    # ax.set_ylabel('Occurence', fontdict=font)

    return


if __name__ == '__main__':
    
    #values dataframe
    df = pd.read_csv('duration_cont.csv',index_col=None)
    df = df_minimize(df)

    #multi plot
    nrow = 1
    ncol = 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*8, nrow*6))    
    

    #fits dataframe eq + val
    for i,eqx in enumerate([double_exp,powerlaw]):
        fits = pd.DataFrame()
        part = pd.DataFrame()
        col_names = []
        for col in df:
            fits[col],_,_ = value_fit(np.array(df[col]),eq=eqx)
            part[col] = df[col]
            col_names.append(col)
        
        ax = axes[i]
        fits['timepoint'] = np.arange(1,len(df)+1)
        part['timepoint'] = np.arange(1,len(df)+1)
        
        #melting fits
        fits = pd.melt(fits,
                       value_vars=col_names,
                       id_vars = ['timepoint'],
                       var_name = 'x',
                       value_name='values')
        #melting data
        part = pd.melt(part,
                       value_vars=col_names,
                       id_vars = ['timepoint'],
                       var_name = 'x',
                       value_name='values')
        #plotting
        sns.lineplot(data=fits,x='timepoint',y='values',
                     ax=axes[i],palette='viridis',hue='x')
        sns.scatterplot(data=part,x='timepoint',y='values',
                        ax=axes[i],palette='viridis',hue='x')
        
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim([0.72, 3.7e3])
        ax.set_ylim([0.5, 1e7])
        
        
        ax.set_xlabel(None)
        ax.set_ylabel(None)
    


