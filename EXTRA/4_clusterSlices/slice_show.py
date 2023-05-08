import numpy as np
import glob
import os
import sys
import time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

def deleteNaN(y):
    """
    delete NaN parts of the input array
    and returns time and respective values.

    """
    
    t = np.arange(len(y))+1
    val = np.array(y)
    
    t = t[~np.isnan(val)]
    val = val[~np.isnan(val)]
    

    return t,val

def double_exp_decay(x,a,b,c,d):
    return a*np.exp(-x*b) + (d)*np.exp(-x*c)

def powerlaw(x,a,b):
	return a*x**(-b)


def exp_decay(x,a,b):
    return a*np.exp(-x*b)

def tri_exp(x,a,b,c,d,e,f):
    return a*np.exp(-x*b) + c*np.exp(-x*d)+e*np.exp(-x*f)

def value_fit(val,eq):
    
    t_range = np.arange(len(val))+1
    
    residual_t = np.zeros([len(val),2])
    
    t,val = deleteNaN(val)
    
    popt, pcov= curve_fit(eq, t, val, maxfev=2000000)
    residuals = (val- eq(t, *popt))/np.max(val)*np.max(t_range)#norm. to max value
    res_norm = residuals/len(val)*len(t_range)#norm. to size
    ss_res_norm = np.sum(res_norm**2)
    # ss_res_norm = ss_res/len(t)
    
    y_fit = eq(t_range, *popt)#full time length
    y_fit[y_fit<1] = np.nan#too small values to be removed
    y_fit[y_fit>np.max(val)*1.2] = np.nan#too big values removed
    
    return y_fit,ss_res_norm

def arr_minimize(arr,method='median'):
    """
    Minimized 1d array by removing repeats,
    according to the given method. 
    ---
    methods: 'median' or 'average'
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

def df_minimize(df):
    """
    minimizes dataframe values by removing repeating values.
    """
    for i in range(len(df.columns)):
        df.iloc[:,i] = arr_minimize(df.iloc[:,i]) #values minimized and returned

    return df


def df_sum_types(df:pd.DataFrame)->pd.DataFrame:
    #finding column names with types
    corenames = [x for x in df.columns if 'core' in x]
    surfnames = [x for x in df.columns if 'surf' in x]
    freenames = [x for x in df.columns if 'free' in x]

    #summing up types in another dataframe
    sums = pd.DataFrame()
    sums['core'] = df[corenames].sum(axis=1)
    sums['surf'] = df[surfnames].sum(axis=1)
    sums['free'] = df[freenames].sum(axis=1)

    return sums

def graphit(df:pd.DataFrame)->plt.Axes:
    
    ax = plt.figure()
    font = {'family': 'Arial',
            'weight': 'bold',
            'size': 14,
            }
    sns.set(rc = {'figure.figsize':(6.6,4.4),
                  'font.weight':'bold',
                  'font.family':'Arial',
                  'font.size':12}
            )
    
    sns.set_style("white")
    sns.scatterplot(data=df.core,color='#2456E5',s=50,edgecolor='black')
    sns.scatterplot(data=df.surf,color='#E57E24',s=50,edgecolor='black')
    sns.scatterplot(data=df.free,color='#454649',s=50,edgecolor='black')
    #lineplot
    sns.lineplot(data=df.corefit,color='#2456E5',linestyle='dashed',alpha=1)
    sns.lineplot(data=df.surffit,color='#E57E24',linestyle='dashed',alpha=1)
    sns.lineplot(data=df.freefit,color='#454649',linestyle='dashed',alpha=1)
    
    plt.yscale('log')
    plt.xscale('log')
    types  = ['Core','Surf','Free']
    plt.legend(types)
    plt.xlabel('Duration (a.u.)',fontdict=font)
    plt.ylabel('Occurence',fontdict=font)
    
    return ax

    
def figures_and_residues(keyword:str,eq)->np.ndarray:
    csv_files = glob.glob('./csv/*.csv')#finding all csv files
    slices = [csv for csv in csv_files if keyword in csv]#cases with keyword

    
    font = {'family': 'Arial',
            'weight': 'bold',
            'size': 13,
            }
    font2 = {'family': 'Calibri',
            'size': 13,
            'color':'#FD4610'
            }
      
    ress_norm = np.zeros([3,len(slices)])
    
    for i,slicee in enumerate(slices):
        
        df = pd.read_csv(slicee)
        
        sums = df_sum_types(df)
        sums.iloc[-1] = np.nan####if full just delete -- data outlier
        sums = df_minimize(sums)

        sums['corefit'],ss_res_core = value_fit(sums.core,eq)
        sums['surffit'],ss_res_surf = value_fit(sums.surf,eq)
        sums['freefit'],ss_res_free = value_fit(sums.free,eq)
        sums.index+=1
        sums[sums==0] = np.nan
        
        #name
        name = slicee[6:-10]
        
        #normalization to max 
        norm = sums.copy()
        norm.core/=sums.core.max()
        norm.corefit/=sums.core.max()
        norm.surf/=sums.surf.max()
        norm.surffit/=sums.surf.max()
        norm.free/=sums.free.max()
        norm.freefit/=sums.free.max()
        
        #graphing and saving
        graph = graphit(norm)

        #saving redisuals
        ress_norm[0,i] = ss_res_core
        ress_norm[1,i] = ss_res_surf
        ress_norm[2,i] = ss_res_free
        #figure and dataframe saving
        fname = name +'.png'
        descr = f'U : {name[:4]}kT\nC : {name[5:7]}µM\nS : {name[-2:].upper()}\nF : TED'
        plt.text(1, 1, descr, fontdict=font2,style='italic')
        plt.savefig('ted'+fname,dpi=200)
        sums.to_csv(name+'.csv',index=True,index_label='time')
        
        
        
        
        
        norm.to_csv('norm_'+name+'.csv',index=True,index_label='time')
    
    return ress_norm

equations = [exp_decay,double_exp_decay,tri_exp,powerlaw]
eqnames = ['ed','ded','ted','pl']
resd = pd.DataFrame()
for eqt,eqname in zip(equations,eqnames):
    resd[eqname]= figures_and_residues('60',eqt).flatten()

plt.figure()

sns.set(style='ticks',
        rc = {'figure.figsize':(9,6),
              'font.weight':'bold',
              'font.family':'Arial',
              'font.size':12
              }
        )

sns.boxplot(data=resd,palette='viridis')
plt.yscale('log')
plt.xlabel('Equations')
plt.ylabel('Sum of Squares of Residuals Normalized')
plt.savefig('eqVSres.png', dpi=200)

    







    
