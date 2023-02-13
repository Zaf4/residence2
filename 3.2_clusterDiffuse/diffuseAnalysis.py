import numpy as np
import pandas as pd
import clusterAnalysisExtra as clust
import os
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from rich.progress import track

def findResidences(arr:np.ndarray)->np.ndarray:
    """
    Finds residence time of particles.

    Parameters
    ----------
    arr : np.ndarray
        State array (full).

    Returns
    -------
    rt : np.mndarray
        Residence Times.

    """
        
    #allocatin an array
    rt = np.zeros([len(arr)])
    for i,row in enumerate(arr):
        a = np.where(row==0)[0]
        if len(a)>0:
            rt[i] = a[0]  
        else:
            rt[i] = len(arr[0])
    

    return rt

def count_bounds(arr:np.ndarray)->np.ndarray:
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
    decay = np.ones([arr.shape[1]],dtype=np.float32)*bound
    
    #finding decay from residences..
    for i,r in enumerate(rt):
        bound-=1
        decay[r:] = bound
        
    decay[decay==0]=np.nan
    
    return decay

def removeDNA(df:pd.DataFrame)->pd.DataFrame:
    return df[df.type>2]

def addAtomIndex(df:pd.DataFrame)->pd.DataFrame:
    if 'atomID' not in df:
        tfatom = len(df)
        df['atomID'] = np.arange(tfatom)+1 #assingning atomID
    return df

def addTFindex(df:pd.DataFrame)->pd.DataFrame:
    if 'tfID' not in df:
        #number of tf = number of tf atoms / 3
        tfmol = int(len(df)/3)
        #tf indexing
        tfIDs = np.repeat(np.arange(tfmol)+1,3)
        df['tfID'] = tfIDs
    return df
    


def findDecayRate(decay:np.ndarray)->float:
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
    
    def exp_decay(x,a,b):
        return a*np.exp(-x*b)
    
    decay = decay[decay>decay[0]/10]#below 10 percent is not counted
    
    timestep_full = np.arange(len(decay))+1
    timestep,decay = timestep_full[decay>0],decay[decay>0]
    
    popt,pcov = curve_fit(exp_decay, timestep, decay, maxfev=2000000)
    coeff,K = popt
    tau = 1/K
    

    return tau

def addClusterLifetimes(df:pd.DataFrame,
                         states:np.ndarray,
                         timestep:int = 1000,
                         duration:int = 2000)->pd.DataFrame:

    #initing taus array
    taus = np.zeros(len(df),dtype = np.float32)
    
    #for each cluster --also nonclusters-- add decay rates for that cluster
    for i in range(int(df.clusterID.max())+1):
        #isolating a cluster
        single_cluster = df[df.clusterID==i] 
        tfIDs = np.unique(single_cluster.tfID)
        atomIDs = np.unique(single_cluster.atomID)
        
        #for only tfs in the given --cluster-- and --timestep--
        partial = states[tfIDs-1,timestep:timestep+duration]
        
        #finding tau of a particular cluster
        decay = count_bounds(partial)
        tau = findDecayRate(decay)
        
        #assigning taus to relevant places
        taus[atomIDs-1] = tau
    
    #add taus array to df
    df['tau'] = taus
    
    return df

def addResidenceTimes(df:pd.DataFrame,
                      states:np.ndarray,
                      timestep:int=1000)->pd.DataFrame:
    #find residences and assing it to their respective tfs
    rt_tf = findResidences(states[:,timestep:])
    rt_atoms = np.repeat(rt_tf,3)
    
    df['residence'] = rt_atoms
    
    return df


def addIndexes(df:pd.DataFrame):
       
    df = removeDNA(df)#removing DNA 
    df = addAtomIndex(df)#atom indexing
    df = addTFindex(df)#TF indexing

    return df


def addClusterSizes(df:pd.DataFrame)->pd.DataFrame:
    #initing taus array
    sizes = np.ones(len(df),dtype = np.int32)
    
    #for each cluster finds its size and assing it to atoms of it
    for i in range(1,int(df.clusterID.max())+1):
        #isolating a cluster
        single_cluster = df[df.clusterID==i]
        size = int(len(single_cluster)/3)
        atomIDs = single_cluster.atomID
        sizes[atomIDs-1] = size
    
    df['clusterSize'] = sizes
    
    return df

def addTimestep(df:pd.DataFrame,timestep:int)->pd.DataFrame:
    ts_arr = np.zeros(len(df),dtype=np.int32)
    ts_arr[:] = timestep
    df['timestep'] = ts_arr
    return df


def addClusterShape(df:pd.DataFrame)->pd.DataFrame:
    
    num_clust = df.clusterID.max()
    
    conformations = np.repeat(['None'+' '*12],len(df))
    for i in np.arange(num_clust)+1:
        partial = df[df.clusterID==i]
        atomIDs = partial.atomID
        coor = np.array(partial[['x','y','z']])
        shape = clust.find_cluster_shape(coor)
        conformations[atomIDs-1] = shape

    conformations = [x.strip() for x in conformations]
    df['conformation'] = conformations
    
    
    return df

def addKT(df:pd.DataFrame,kT:float)->pd.DataFrame:
    kts = np.zeros(len(df))
    kts[:] = kT
    df['kT'] = kts
    
    return df
    
    
def generateDF(dumpfile:os.PathLike='./data/targetDUMP.npy',
               statesfile:os.PathLike='./data/full.npy',
               timestep=2100,kT:float=1.23):
        
    #load the dump file 3D, 0=timestep,1=atoms,2=attributes (type,x,y,z)
    dump = np.load(dumpfile)
    df = clust.cluster_single_frame(dump[timestep])

    #load residence states rows=proteins, columns=time
    states = np.load(statesfile)
    stateL = states[1::2].astype(int)
    stateR = states[0::2].astype(int)
    stateU = stateL+stateR
    stateU[stateU==1] = 0
    stateU[stateU==2] = 1
    states = stateU.astype(bool)
    
    #adding indexes
    df = addIndexes(df)
    #adding mean life times (clusters)
    df = addClusterLifetimes(df, states, timestep=timestep,duration=300)
    #adding residence times (all atoms)
    df = addResidenceTimes(df, states,timestep=timestep)
    #adding clustersizses
    df = addClusterSizes(df)
    #adding timesteps
    df = addTimestep(df, timestep=timestep)
    #add cluster shape
    df = addClusterShape(df)
    #add kTs (energies)
    df = addKT(df, kT)
    
    return df

def multiTimeDF(timesteps:np.ndarray,**kwargs)->pd.DataFrame:
    
    #initing empty dataframe
    df = pd.DataFrame()
    for timestep in track(timesteps):
        ts = generateDF(timestep=timestep,**kwargs)
        df = pd.concat([df,ts])
        
    return df

def makePlot(df:pd.DataFrame,*args,**kwargs)->plt.Axes:
    
    
    # font = 1
    sns.set_theme(style='ticks',
        rc = {
              'font.weight':'light',
              'font.size':14,
              'font.family':'Arial',
              'ytick.minor.size':'0',
              'ytick.major.size':'10',
              'xtick.major.size':'10'
              
              }
        )
    
    df = df[df.clusterID>0]
    max_time = df.timestep.min()
    N = len(np.unique(df.timestep))
    fig,ax = plt.subplots(1,2,figsize=(16,4.5),
                          gridspec_kw={'width_ratios': [2.5, 1]})
    #showing the system -------------SCATTER-----------------------------------
    sns.scatterplot(df[df.timestep==max_time],x='x',y='y',
                    hue='tau',palette='Purples',
                    edgecolor='k',ax=ax[0])
    ax[0].annotate(xy=(0.05,0.1),
                   text=f't = {max_time} a.u',fontweight='bold',
                   xycoords='axes fraction')
    ax[0].set_xlim([-90,90])
    ax[0].set_ylim([-30,30])
    #adding color bar
    norm = plt.Normalize(df.tau.min(), df.tau.max())
    purples = plt.cm.ScalarMappable(cmap="Purples",norm=norm)
    purples.set_array([])
    ax[0].get_legend().remove()
    ax[0].figure.colorbar(purples,ax=ax[0],location='right',
                          shrink=1,label='Mean Lifetime')
    

    #regression  -----------------REGRESSION-----------------------------------
    sns.regplot(df,x='clusterSize',y='tau',
                ax=ax[1],color='red')
    #finding correlation coeff
    r,p = stats.pearsonr(df['clusterSize'],y=df['tau'])
    ax[1].annotate(xy=(0.05,0.90),
                   text=f'r = {r:.2f}',fontweight='bold',
                   xycoords='axes fraction')
    ax[1].annotate(xy=(0.85,0.05),
                   text=f'N = {N}',fontweight='bold',
                   xycoords='axes fraction')
    ax[1].set_ylabel('Mean Lifetime')
    ax[1].set_xlabel('Cluster Size')
    fig.tight_layout()
    
    return ax


def makePlotMulti(dfs:list,*args,**kwargs)->plt.Axes:
    
    
    sns.set_theme(style='ticks',
        rc = {
              'font.weight':'light',
              'font.size':14,
              'font.family':'Arial',
              'ytick.minor.size':'0',
              'ytick.major.size':'10',
              'xtick.major.size':'10'
              
              }
        )
    
    num_rows = len(dfs)
    fig,ax = plt.subplots(num_rows,2,figsize=(16,4.5*num_rows),
                          gridspec_kw={'width_ratios': [2.5, 1]})
    
    for row,df in enumerate(dfs):
    
        df = df[df.clusterID>0]
        max_time = df.timestep.min()
        N = len(np.unique(df.timestep))
    
        #showing the system -------------SCATTER-----------------------------------
        sns.scatterplot(df[df.timestep==max_time],x='x',y='y',
                        hue='tau',palette='Purples',
                        edgecolor='k',ax=ax[row,0])
        ax[row,0].annotate(xy=(0.05,0.1),
                       text=f't = {max_time} a.u',fontweight='bold',
                       xycoords='axes fraction')
        ax[row,0].set_xlim([-90,90])
        ax[row,0].set_ylim([-30,30])
        #adding color bar
        norm = plt.Normalize(df.tau.min(), df.tau.max())
        purples = plt.cm.ScalarMappable(cmap="Purples",norm=norm)
        purples.set_array([])
        ax[row,0].get_legend().remove()
        ax[row,0].figure.colorbar(purples,ax=ax[row,0],location='right',
                              shrink=1,label='Mean Lifetime')
        
    
        #regression  -----------------REGRESSION-----------------------------------
        sns.regplot(df,x='clusterSize',y='tau',
                    ax=ax[row,1],color='red')
        #finding correlation coeff
        r,p = stats.pearsonr(df['clusterSize'],y=df['tau'])
        ax[row,1].annotate(xy=(0.05,0.90),
                       text=f'r = {r:.2f}',fontweight='bold',
                       xycoords='axes fraction')
        ax[row,1].annotate(xy=(0.85,0.05),
                       text=f'N = {N}',fontweight='bold',
                       xycoords='axes fraction')
        ax[row,1].set_ylabel('Mean Lifetime')
        ax[row,1].set_xlabel('Cluster Size')
    
        fig.tight_layout()
        
    return ax



if __name__ == '__main__':
    """
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
    """
    timesteps = np.random.random_integers(5000,22000,size=40)
    
    #for server
    flocs = '/home/gottar/5x10t/kt/60'
    saveloc = '/home/gottar/5x10t/graphs'
    kts = ['2.80','3.00','3.50','4.00']
    
    #file list
    dumpfiles = [flocs.replace('kt',kt)+'/dump.npy' for kt in kts]
    statesfiles = [flocs.replace('kt',kt)+'/full.npy' for kt in kts]
    
    #generating dfs
    df280 = multiTimeDF(timesteps,dumpfile=dumpfiles[0],statesfile=statesfiles[0],kT=float(kts[0]))
    df300 = multiTimeDF(timesteps,dumpfile=dumpfiles[1],statesfile=statesfiles[1],kT=float(kts[1]))
    df350 = multiTimeDF(timesteps,dumpfile=dumpfiles[2],statesfile=statesfiles[2],kT=float(kts[2]))
    df400 = multiTimeDF(timesteps,dumpfile=dumpfiles[3],statesfile=statesfiles[3],kT=float(kts[3]))
    #save dfs for further uses
    df280.to_csv(f'{saveloc}/280.csv',index=False)
    df300.to_csv(f'{saveloc}/300.csv',index=False)
    df350.to_csv(f'{saveloc}/350.csv',index=False)
    df400.to_csv(f'{saveloc}/400.csv',index=False)
    
    #plotting and saving the plot
    ax = makePlotMulti([df280,df300,df350,df400])
    plt.savefig(f'{saveloc}/multi.png', dpi=400)
    """
    df280 = pd.read_csv('./data/280.csv', index_col=False)
    df300 = pd.read_csv('./data/300.csv', index_col=False)
    df350 = pd.read_csv('./data/350.csv', index_col=False)
    df400 = pd.read_csv('./data/400.csv', index_col=False)
    
    ax = makePlotMulti([df280,df300,df350,df400])
    plt.savefig('./graphs/multi.png', dpi=400)
    
    