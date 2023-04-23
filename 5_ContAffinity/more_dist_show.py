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

    
    popt, pcov= curve_fit(eq, t, val, maxfev=20000000,sigma=1/t**4)
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

df = pd.read_csv('duration_cont.csv',index_col=None)

df = df_minimize(df)
values = np.array(df.uniform_350)


fits = pd.DataFrame()


for eqx in [double_exp,powerlaw,st_exp]:
    fits[eqx.__name__],_,_ = value_fit(values,eq=eqx)

df.index = np.arange(len(df))+1
fits.index+=1
#sns.scatterplot(df,x=df.index,y='gaus_350',palette='magma_r')
ax = plt.figure(figsize=(5.6,4.2))
sns.set_palette('viridis')
sns.scatterplot(df)
"""
sns.scatterplot(df,x=df.index,y='uniform_350')
sns.scatterplot(df,x=df.index,y='gaus_350')
sns.scatterplot(df,x=df.index,y='3.50_60')"""
#sns.lineplot(fits,palette='Purples_r')

plt.xscale('log')
plt.yscale('log')
#plt.show()
plt.savefig('image.png',dpi=400)