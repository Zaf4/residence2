import numpy as np
import pandas as pd

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
    print(x1s)
    print(x1s[order])
    
    popt[0::2] = coeffs[order]
    popt[1::2] = expos[order]
    
    return popt

if __name__ == '__main__':
    ex = pd.read_csv('./data/exponents.csv',index_col=None,
                     encoding= 'unicode_escape')
    
    popts = [x for x in ex.columns if ('tau' in x or 'coeff' in x)]
    popt = ex.iloc[90][popts]
    
    popt = np.array(popt)
    popt2=order_by_pair(popt,rate2tau=0)

