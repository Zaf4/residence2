import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#clustering algorithm
def find_cluster(atoms:np.ndarray,
                 threshold:float=2.1,
                 min_size:int=5)->np.ndarray:
    """
        
    Takes an array(atoms) with the coordinates of atoms
    clusters atoms that are close to each other less
    than threshold.

    Returns cluster_arr 
    row indexes: unique clusters
    column indexes: atoms, marked 1 if associated.
    
    Parameters
    ----------
    atoms : numpy.ndarray
          atoms array. 
    threshold : float
        Threshold distance for neighbours. Default is 2.1.
    min_size : int
        minimum size of molecule collection to be considered a cluster.
        Default is 5.
        
    Returns
    -------
    cluster_array : numpy.ndarray
        rows->cluster index-1
        columns->every protein is marked 0(not part of) or 1 (part of) that
        particular row or cluster.

    
    """
    
    num = len(atoms)
    groups = np.zeros((num,num),dtype='float32')
    
    #neighbors lists for each atom is found
    for i in range(num):
        #euclidian distance of distances in each (x,y,z) direction
        distance = np.linalg.norm(np.subtract(atoms,atoms[i]),axis=1)
        neighbors = distance<threshold
        neighbors[i] = False
        groups[i] = neighbors
    
    #neighbor lists are united repeats are deleted
    arr = groups.copy()
    jump = []#atoms that are assigned to a cluster to be jumped
    for j in range(num):
        if j in jump:
            continue
        index, = np.where(arr[j] > 0)
        
        for k in index:
            if np.sum(arr[j]) > 0 and k>j:
                arr[k] += arr[j]
                arr[j] = np.zeros(num,dtype='float32')
                ind, = np.where(arr[k] != 0)
                if np.sum(arr[ind]) == 0:
                    jump.append(k)
    
    
    c, = np.where(np.sum(arr,axis=1)>0)
    cluster_arr = arr[c].copy()
    
    cluster_arr[cluster_arr>1] = 1
    cluster_size = np.sum(cluster_arr,axis=1)/3 #3 atoms per protein, 
    cluster_arr = cluster_arr[cluster_size>min_size]
    return cluster_arr

def cluster_id(cluster_arr:np.ndarray,n:int=0)->np.ndarray:
    """
    
    from cluster_arr of clusters and atoms marked to be contained,
    returns a 1D numpy array that marks atoms with their respective
    cluster ids. 

    if there are not-to-be-clustered atoms with number (n) they are 
    initiated as 0 before the cluster atoms.
    

    Parameters
    ----------
    cluster_arr : numpy.ndarray
        rows->cluster index-1
        columns->every protein is marked 0(not part of) or 1 (part of) that
        particular row or cluster.
    n : int, optional
        number of not-to-be-clustered atoms. The default is 0.

    Returns
    -------
    cluster : numpy.ndarray
        1D array with the size of n+cluster atoms.

    """
    if len(cluster_arr)>0:
        width = len(cluster_arr[0])+n #cluster atoms + other atoms
        cluster = np.zeros([width])
    
        for j,c in enumerate(cluster_arr):
            cluster_num = j+1 #giving ids to the cluster
            cluster_atoms =  np.where(c==1)[0] #indexes of atoms
            index_in_df = n+cluster_atoms  #indexes in the df (n, number of not-to-be-clustered atoms)
            cluster[index_in_df] = cluster_num
    else:
        return

    return cluster


def cluster_single_df(frame:pd.DataFrame)->pd.DataFrame:
    """
    Takes single time point as argument and returns a pandas.DataFrame with
    cluster IDs for each particle

    Parameters
    ----------
    frame : np.ndarray
        takes a single time point Frame.
        
    min_size : int, optional
        min_size for chunk of molecules to form clusters. 
        The default is 12.
        
    clusterit : bool, optional
        true->add cluster column to dataframe
        false->do not cluster column to dataframe

    Returns
    -------
    ss : pd.DataFrame
        DataFrame with cluster IDs (if cluster==True) for each particle.
   

    """
    
    ss = frame[(frame.type!=6) & (frame.type>2)].copy()
    n_dna = len(ss[ss.type<3])
    tf_coor = np.array(ss[['x','y','z']])

    cluster = find_cluster(tf_coor,min_size=12)#returns cluster array
    cluster_index = cluster_id(cluster,n=n_dna)
    ss['clusterID'] = cluster_index
    
    return ss

def data2df(file:os.PathLike)->pd.DataFrame:
    """
    Turns LAMMPS data file into DataFrame 
    via retrieving atoms and their information.
    with columns:'atomID','type','xco','yco','zco'
    

    Parameters
    ----------
    file : os.PathLike
        path/to/file.

    Returns
    -------
    df_atoms : pd.DataFrame
        DataFrame containing information of the datafile.

    """
    #readinf the fike
    f = open(file,'r')
    lines = f.readlines()
    f.close()
    
    #finding num of atoms

    num_atoms = int(lines[2].split(' ')[0])
    
    #finding beginning and the end of the atoms
    for i,l in enumerate(lines):
        if l[0:5] == 'Atoms':
            start = i+2
        
    
    #finding start and end
    end = start+num_atoms
    
    atom_lines = lines[start:end]
    coor = [a.split(' ') for a in atom_lines] #string to list
    coor = [c for c in coor if c != ['\n']] #remove empty lines if present
    # coor = [c for c in coor if c[1] != '6'] #remove capsid
    
    rows = len(coor)
    cols = 5
    col = [0,1,3,4,5] #dont take mol-type
    
    #initing an array with intented size
    atoms = np.zeros([rows,cols])
    
    #assigning values to the array
    for c in coor:
        for i,j in enumerate(col):
            atomID = int(c[0])-1
            atoms[atomID,i] = float(c[j])
            
    #array to dataframe conversion with given column names
    names = ['atomID','type','x','y','z']
    df_atoms = pd.DataFrame(atoms, columns = names)
    
    return df_atoms


def _modify_scatter(func):
    def modifier(*args,**kwargs):

        ax = func(*args,**kwargs)
        
        ax.set(xticklabels=[],yticklabels=[])
        ax.legend([])
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_ylim([-31,31])
        ax.set_xlim([-92,92])
    
        return ax
    return modifier
        


@_modify_scatter
def system_scatter(coor:pd.DataFrame,axes,row,col,**kwargs):
    
    font = {'family': 'sans-serif',
            'weight': 'light',
            'style' : 'italic',
            'size': 16,
            }
    
    #seperate capsid from others
    coor = addRadius2df(coor)
    capsid = coor[coor.type==6]
    coor = coor[coor.type!=6]
    #reduce capsid
    capsid = capsid[capsid.z>-1.2]
    capsid = capsid[capsid.z<1.2] 


    #selecting ax
    ax = axes[row,col]
    #graphing
    sns.scatterplot(data=coor,**kwargs,size='radius',
                    palette='Blues',edgecolor='k',linewidth=0.3,ax=ax)
    sns.scatterplot(data=capsid,**kwargs,palette='mako',
                    edgecolor='k',linewidth=0.4,ax=ax)
    
    # ax.annotate(x=-90,y=27,text=f'Rg={Rg(coor):.1f}',fontdict=font)
    
    return ax

@_modify_scatter
def protein_scatter(coor:pd.DataFrame,axes,row,col,**kwargs):
    
    #seperate capsid from others
    coor = addRadius2df(coor)
    capsid = coor[coor.type==6]
    coor = coor[(coor.type>2) & (coor.type<6)]
    #reduce capsid
    capsid = capsid[capsid.z>-1.2]
    capsid = capsid[capsid.z<1.2] 
    #selecting ax
    ax = axes[row,col]
    #graphing
    
    sns.scatterplot(data=coor,**kwargs,size='radius', palette = 'Paired',
                         edgecolor='k',linewidth=0.3,ax=ax)
    sns.scatterplot(data=capsid,**kwargs,palette='mako',
                         edgecolor='k',linewidth=0.4,ax=ax)

    
    return ax

@_modify_scatter
def cluster_scatter(coor:pd.DataFrame,axes,row,col,**kwargs):
    
    #seperate capsid from others
    coor = addRadius2df(coor)
    capsid = coor[coor.type==6]
    clusterFrame = cluster_single_df(coor)
    tfs = clusterFrame[clusterFrame.type>2]
    # coor = coor[(coor.type>2) & (coor.type<6)]
    #reduce capsid
    capsid = capsid[capsid.z>-1.2]
    capsid = capsid[capsid.z<1.2] 
    #selecting ax
    ax = axes[row,col]
    #graphing
    sns.scatterplot(data=tfs[tfs.clusterID>0],**kwargs,size='radius', 
                    palette = 'viridis',hue='clusterID',
                    edgecolor='k',linewidth=0.3,ax=ax)
    sns.scatterplot(data=capsid,**kwargs,palette='mako',
                    edgecolor='k',linewidth=0.4,hue='type',ax=ax)
    ax.set_axis_off()
    
    return ax

def Rg(df)->float:
    
    dna = df[df.type<3]
    coor = np.array(dna[['x','y','z']])
    average = np.average(coor,axis=0)#center of mass
    coor_norm = np.subtract(coor,average)#distances from CoM
    r2 = np.sum(coor_norm**2,axis=1)
    rg2 = np.average(r2)
    rg = np.sqrt(rg2)
    
    return rg
    
def addRadius2df(df:pd.DataFrame)->pd.DataFrame:
    
    r = np.zeros(len(df))
    r[df.type==1] = 50
    r[df.type==2] = 50
    r[df.type==3] = 20
    r[df.type==5] = 50
    r[df.type==6] = 20
    
    df['radius'] = r
    
    return df
    
def createFigures(cases = [100,280,300,350,400]):
    
    nrow = len(cases)
    sns.set(style='white',
            rc = {
                    'font.weight':'light',
                    'font.family':'sans-serif',
                    'axes.spines.top':'False',
                    'axes.spines.right':'False',
                    'axes.spines.bottom':'False',
                    'axes.spines.left':'False',
                    'ytick.minor.size':'0',
                    'ytick.major.size':'0',
                    'xtick.major.size':'0',
                    'legend.frameon':False,
                    
                    }
            )
    fig, axes = plt.subplots(nrows=nrow,ncols=3,figsize=(3*10,nrow*4))
    
    
    for i,case in enumerate(cases):
        coor = data2df(f'./data/data{case}.extra')
        if os.path.exists('./figures') == False:
            os.mkdir('./figures')
            
        
        #entire system
        system_scatter(coor,axes,row=i,col=0,x='x',y='y',hue='type')

        #proteins alone
        protein_scatter(coor,axes,row=i,col=1,x='x',y='y',hue='type')

        #only clusters
        cluster_scatter(coor,axes,row=i,col=2,x='x',y='y')

    
    return fig


if __name__ == '__main__':
    #changing working directory to current directory name
    os.chdir(os.path.dirname(__file__))

    fig = createFigures()
    plt.tight_layout()

    # plt.annotate('A',xycoords='figure fraction', xy = (0.01,0.98),fontsize=64,color='black')
    # plt.annotate('B',xycoords='figure fraction', xy = (0.34,0.98),fontsize=64,color='black')
    # plt.annotate('C',xycoords='figure fraction', xy = (0.67,0.98),fontsize=64,color='black')
    
    # fig.savefig('../Figures/fig3.pdf', transparent=True, bbox_inches='tight')
    fig.savefig('../Figures/fig3.png',transparent=True, bbox_inches='tight',dpi=96)

    
