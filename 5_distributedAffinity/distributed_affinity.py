import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import math
import os
from scipy.optimize import curve_fit
from scipy import special

# suppress the warning
warnings.filterwarnings('ignore')
pd.set_option('mode.chained_assignment', None)

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

def exp_decay(x,a,b):
    return a*np.exp(-x*b)

def tri_exp(x,a,b,c,d,e,f):
    return a*np.exp(-x*b) + c*np.exp(-x*d)+e*np.exp(-x*f)

def quad_exp(x,a,b,c,d,e,f,g,h):
    return a*np.exp(-x*b) + c*np.exp(-x*d)+e*np.exp(-x*f)+g*np.exp(-x*h)

def penta_exp(x,a,b,c,d,e,f,g,h,i,j):
    return a*np.exp(-x*b) + c*np.exp(-x*d)+e*np.exp(-x*f)+g*np.exp(-x*h)+i*np.exp(-x*j)

def value_fit(val:np.ndarray,
              eq:callable,sigma_w:bool = False)->tuple[np.ndarray,np.ndarray,tuple]:
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
    t = t.astype(np.float64)
    val = val.astype(np.float64)

    if sigma_w:
        sigma = t**-10
        popt, pcov= curve_fit(eq, t, val, 
                              maxfev=20000000,
                              sigma=sigma)

    else:
        popt, pcov= curve_fit(eq, t, val,
                              maxfev=20000000)
    
    residuals = (val- eq(t, *popt))
    ress_sumofsqr =np.sum(residuals**2)
    ss_res_norm = ress_sumofsqr/len(val)
    # ss_res_norm = ss_res/len(t)
    
    y_fit = eq(t_range, *popt)#full time length
    y_fit[y_fit<1] = np.nan#too small values to be removed
    y_fit[y_fit>np.max(val)*2] = np.nan#too big values removed
    
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
        elif method == 'max': 
            mid = int(np.max(positions))
        elif method == 'min': 
            mid = int(np.min(positions))
        
        arr1[positions] = np.nan
        arr1[mid] = s #mid value is kept
        
    return arr1

def df_minimize(df:pd.DataFrame,**kwargs)->pd.DataFrame:
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
        df.iloc[:,i] = arr_minimize(df.iloc[:,i],**kwargs) #values minimized and returned

    return df

def scatterit_multi(df: pd.DataFrame, fits: pd.DataFrame,
                    i:int, j:int, axes:plt.Axes,
                    palette: str, **kwargs)->None:
    """subplots of scatters

    Parameters
    ----------
    df : pd.DataFrame
        distributions
    fits : pd.DataFrame
        fits
    i : int
        row
    j : int
        column
    axes : plt.Axes
        axes
    palette : str
        color palette of plots
    """
    
    #preprocess raw data and fits for graph------------------------------------
    cols = [col for col in df]
    df['timestep'] = np.arange(len(df))+1
    fits['timestep'] = np.arange(len(df))+1
    
    fits = pd.melt(fits,
                   id_vars=['timestep'],
                   value_vars=cols,
                   var_name='case',
                   value_name='value')
    
    df = pd.melt(df,
                 id_vars=['timestep'],
                 value_vars=cols,
                 var_name='case',
                 value_name='value')
    
    ax = axes[i,j]

    # resetting indexes
    df.index = np.arange(len(df))+1
    fits.index = np.arange(len(df))+1

    #font settings
    font = {'family': 'Sans Serif',
            'weight': 'light',
            'size': 20,
            }
    
    #weight calculation to prevent overcrowding
    ts = np.array(df.timestep)
    # weights = ((1/(ts*ts[::-1]))*10**5)**2
    tmax = df.timestep.max()
    weights = ts**2-tmax*ts+(tmax/2)**2
    #scattter plot (Data)------------------------------------------------------
    sns.scatterplot(data=df.sample(frac=0.25,random_state=42,weights=weights),
                    x='timestep',
                    y='value',
                    palette=palette,
                    hue='case',
                    hue_order=cols,
                    alpha=0.7,
                    s=300,
                    edgecolor='white', 
                    linewidth=1.8,
                    # legend=False,
                    ax=ax,
                    **kwargs)
    
    #Line plot (Fits)----------------------------------------------------------
    sns.lineplot(data=fits,
                 x='timestep',
                 y='value',
                 palette=palette,
                 hue='case',
                 ax=ax,
                 linewidth=7,
                 linestyle='dashed',
                 legend=False,
                 alpha=1, 
                 **kwargs)

    #graph settings------------------------------------------------------------
        
    ax.tick_params(axis='both',labelsize=24)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.72, 3.7e3])
    ax.set_ylim([0.5, 1e7])
    #ax.get_legend().remove()
    if j == 0:
        ax.legend(markerscale=4,fontsize=21)
    else:
        ax.get_legend().remove()

    return


if __name__ == '__main__':
    #changing working directory to current directory name
    os.chdir(os.path.dirname(__file__))
    
    #values dataframe
    df = pd.read_csv('./data/duration_cont.csv',index_col=None)
    df = df_minimize(df,method='median')
    df[df==0] = np.nan
    ts = np.arange(len(df))+1
    df['timestep'] = ts
    
    df.rename(columns = {'uniform_250':'1-4kT Uniform', 'uniform_350':'2-5kT Uniform',
                        'gaus_250':'1-4kT Normal','gaus_350':'2-5kT Normal'}, inplace = True)

    #reference line
    ref = df[['timestep','3.50_60']]
    #weight calculation to prevent overcrowding
    tmax = df.timestep.max()
    weights = ts**2-tmax*ts+(tmax/2)**2
    
    
    sns.set_theme(style='ticks',
                  rc={
                  'font.weight': 'light',
                  'font.family': 'sans-serif',
                  'axes.spines.top': 'False',
                  'axes.spines.right': 'False',
                  'ytick.minor.size': '0',
                  'xtick.minor.size': '0',
                  'ytick.major.size': '10',
                  'xtick.major.size': '10',
                  'legend.frameon': False,
    
                     }
                  )
    #multi plot
    nrow = 2
    ncol = 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*7, nrow*6))    
    fig.supxlabel('Duration (a.u.)', fontsize=28,fontweight='light')
    # fig.supylabel('Occurence (n)', fontsize=28,fontweight='light',x=0.01) 
    #Unifrom vs Gaussian distro
    uniform = ['1-4kT Uniform', '2-5kT Uniform',]
    gaus = ['1-4kT Normal','2-5kT Normal']
    
    palettes= ['RdPu','mako_r']
    #fits dataframe eq + val
    for i,eqx in enumerate([tri_exp,powerlaw,powerlaw]):
        fits = pd.DataFrame()
        part = pd.DataFrame()
        col_names = []

        #creating fits and partial data df
        for col in df:
            if i == 2:
                
                fits[col],_,_ = value_fit(np.array(df[col]),eq=eqx,sigma_w=True)#fitting part
            else:
                
                fits[col],_,_ = value_fit(np.array(df[col]),eq=eqx)#fitting part
            part[col] = df[col]
            col_names.append(col)
            
        for j,col_type in enumerate([uniform,gaus]):
            
            ax = axes[j,i]
            
            scatterit_multi(part[col_type],
                            fits,
                            i=j,
                            j=i,
                            axes=axes,
                            palette=palettes[j])
            
            sns.lineplot(data=ref,
                            x='timestep',
                            y='3.50_60',
                            ax=ax,
                            color='k',
                            linewidth=3,
                            legend=False,
                            )
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            
    fig.tight_layout(w_pad=1,h_pad=1)
    # plt.annotate('A',xycoords='figure fraction', xy = (0.01,0.95),fontsize=48)
    # plt.annotate('B',xycoords='figure fraction', xy = (0.34,0.95),fontsize=48)
    # plt.annotate('C',xycoords='figure fraction', xy = (0.67,0.95),fontsize=48)
    fig.savefig('../Figures/fig6.pdf', transparent=True)
    fig.savefig('../Figures/fig6.png', dpi=300, transparent=True)
