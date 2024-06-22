import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erfc
import warnings
import os

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


def exp_decay(x,a,b):
    return a*np.exp(-x*b)

def double_exp(x,a,b,c,d):
    return a*np.exp(-x*b) + (d)*np.exp(-x*c)

def tri_exp(x,a,b,c,d,e,f):
    return a*np.exp(-x*b) + c*np.exp(-x*d)+e*np.exp(-x*f)

def quad_exp(x,a,b,c,d,e,f,g,h):
    return a*np.exp(-x*b) + c*np.exp(-x*d)+e*np.exp(-x*f)+g*np.exp(-x*h)

def penta_exp(x,a,b,c,d,e,f,g,h,i,j):
    return a*np.exp(-x*b) + c*np.exp(-x*d)+e*np.exp(-x*f)+g*np.exp(-x*h)+i*np.exp(-x*j)

def powerlaw(x,a,b):
	return a*x**(-b)

def erfc_exp(x,a,b):
    return a*erfc(b*x)


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
        fit parameters
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

def order_by_pair(popt:np.ndarray,rate2tau:bool=True)->np.ndarray:
    """
    

    Parameters
    ----------
    popt : np.ndarray
        Input dublets (coeff1-exp1).

    Returns
    -------
    popt : np.ndarray
        Output-dublets.

    """

    coeffs = popt.copy()[0::2]
    expos = popt.copy()[1::2]
    x1s = np.zeros(len(expos))
    if rate2tau:
        expos = 1/expos
        
    for i,(tau,coeff) in enumerate(zip(expos,coeffs)):
        x1s[i] = -np.log(1/coeff)*tau
        
    
    order = np.argsort(x1s)[::-1]
    
    popt[0::2] = coeffs[order]
    popt[1::2] = expos[order]
    
    return popt


def equation_fit_save(datafile:os.PathLike)->None:
    """
    From the durations-occurences dataframe produces fits for given equations
    and saves the files.
    Also saves the minimized data file and exponents for the fits.
    

    Parameters
    ----------
    datafile : os.PathLike
        Path/to/datafile.

    Returns
    -------
    None
 
    """
    
    #durations file is read as the main raw data -->preprocessing
    durations = pd.read_csv(datafile,index_col=None)
    durations[durations == 0] = np.nan 
    durations = df_minimize(durations)
    durations.to_csv('./data/durations_minimized.csv',index=False)
    
    #names and functions are listed for the for loop
    eqnames = ['ED','DED','TED','QED','PED','Powerlaw']
    equations = [exp_decay,double_exp,tri_exp,quad_exp,penta_exp,powerlaw]

    #initing residuals dataframe
    exponents = 'equation,energy,concentration,'
    exponents+= ','.join([f'coeff{i+1},tau{i+1}' for i in range(5)])+'\n'
    residues = pd.DataFrame()
    for name,equation in zip(eqnames,equations):

        fits = pd.DataFrame() #a fit data frame is opened for each equation
        ress = np.zeros([20]) #array to store residual values
        
        #for each column of durations file
        for i,d in enumerate(durations):
            #fitting to equation is done
            fits[d],resid_sum_sqr,popt=value_fit(np.array(durations[d]),eq=equation)
            ress[i] = resid_sum_sqr #collecting residuals
            
            #saving fit coefficents to a file
            exponents+= f'{name},{d[:4]}kT,{d[-2:]}µM,'
            popt = order_by_pair(popt)
            for expo in popt:
                exponents+=f'{expo}'
                if expo != popt[-1]:
                    exponents+=','
            exponents+='\n'
    
        residues[name] = ress #residuals appended
        
        #fits saved to file with their shorthened names + .csv
        fits.to_csv(f'./data/{name}.csv',index=False)
        #residuals saved as .csv 
        residues.index = durations.columns
        residues.to_csv('./data/residuals.csv')  
     
    with open('./data/exponents.csv','w') as file_exp:
        file_exp.write(exponents)
                    
    return 

    
if __name__ == '__main__':
    #changing working directory to current directory name
    os.chdir(os.path.dirname(__file__))

    equation_fit_save('./data/durations.csv')