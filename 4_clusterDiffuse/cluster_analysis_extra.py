import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from rich.progress import track



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
    
    
    width = len(cluster_arr[0])+n #cluster atoms + other atoms
    cluster = np.zeros([width])

    for j,c in enumerate(cluster_arr):
        cluster_num = j+1 #giving ids to the cluster
        cluster_atoms =  np.where(c==1)[0] #indexes of atoms
        index_in_df = n+cluster_atoms  #indexes in the df (n, number of not-to-be-clustered atoms)
        cluster[index_in_df] = cluster_num

    return cluster


def cluster_single_frame(frame:np.ndarray,
                         min_size:int=20,
                         clusterit:bool=True)->pd.DataFrame:
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

    ss = pd.DataFrame(frame,columns=['type','x','y','z'])
    n_dna = len(ss[ss.type<3])
    tf = ss[ss.type>2]
    tf_coor = np.array(tf)[:,1:]

    
    if clusterit:
        cluster = find_cluster(tf_coor,min_size=min_size)#returns cluster array
        if len(cluster) == 0:
            ss['clusterID'] = np.zeros(len(ss)).astype(np.int32)
        else:
            cluster_index = cluster_id(cluster,n=n_dna)
            ss['clusterID'] = cluster_index.astype(np.int32)
    
    return ss

def find_cluster_shape(coor:np.ndarray)->str:

    average = np.average(coor,axis=0)#to center the cluster
    x = average[0]
    y = average[1]
    z = average[2]
    
    #rg x-axis
    rgxx = np.sqrt(np.sum(abs((coor[:,0]-x)*(coor[:,0]-x))))
    rgxy = np.sqrt(np.sum(abs((coor[:,0]-x)*(coor[:,1]-y))))
    rgxz = np.sqrt(np.sum(abs((coor[:,0]-x)*(coor[:,2]-z))))
    #rg y-axis
    rgyy = np.sqrt(np.sum(abs((coor[:,1]-y)*(coor[:,1]-y))))
    rgyz = np.sqrt(np.sum(abs((coor[:,1]-y)*(coor[:,2]-z))))
    #rg z-axis
    rgzz = np.sqrt(np.sum(abs((coor[:,2]-z)*(coor[:,2]-z))))
    
    rgM = [[rgxx,rgxy,rgxz],
           [rgxy,rgyy,rgyz],
           [rgxz,rgyz,rgzz]]
    
    rgM = np.square(rgM)
    
    eigV,eigM1 = np.linalg.eig(rgM)
    eigV = np.linalg.eigvals(rgM)
    
    shape = 'globular'
    if np.max(eigV)>20*np.min(eigV):
        shape = 'filamentous'
    elif np.max(eigV)>12*np.min(eigV):
        shape = 'semi-filamentous'

    return shape

def generate_cluster_info_table(frame:np.ndarray,timestep:int=0,
                                simple:bool=True)->pd.DataFrame:
    """
    Generates a cluster table (pd.DataFrame) that summarizes each cluster
    for the given frame.

    Parameters
    ----------
    frame : np.ndarray
        Single time point or frame of the 3d array (dump).

    Returns
    -------
    
    cluster_frame : pd.DataFrame
        Summary table of cluster
        
    eigM : np.ndarray
        eigenvalue vector for each cluster in a 2d array.

    """
    
    ss = cluster_single_frame(frame,min_size = 12)
    n_cluster = int(ss.clusterID.max())
    liste = []
    
    eigM = np.zeros([n_cluster,3])

    for index in range(1,n_cluster+1):
        cluster_sub_df = ss[ss.clusterID == index]
        coor = np.array(cluster_sub_df[['x','y','z']])
        average = np.average(coor,axis=0)#to center the cluster
        coor_norm = np.subtract(coor,average)#cluster centerd
        std_xyz = np.std(coor_norm,axis=0)#
        rg_xyz = np.average(coor_norm**2,axis=0)
        rg_std = np.std(rg_xyz)
        rg_avg = np.average(rg_xyz)
        rg_min = np.min(rg_xyz)
        rg_max = np.max(rg_xyz)
        mm_avg = (rg_max+rg_min)/2
        
        x = average[0]
        y = average[1]
        z = average[2]
        
        #rg x-axis
        rgxx = np.sqrt(np.sum(abs((coor[:,0]-x)*(coor[:,0]-x))))
        rgxy = np.sqrt(np.sum(abs((coor[:,0]-x)*(coor[:,1]-y))))
        rgxz = np.sqrt(np.sum(abs((coor[:,0]-x)*(coor[:,2]-z))))
        #rg y-axis
        rgyy = np.sqrt(np.sum(abs((coor[:,1]-y)*(coor[:,1]-y))))
        rgyz = np.sqrt(np.sum(abs((coor[:,1]-y)*(coor[:,2]-z))))
        #rg z-axis
        rgzz = np.sqrt(np.sum(abs((coor[:,2]-z)*(coor[:,2]-z))))
        
        rgM = [[rgxx,rgxy,rgxz],
               [rgxy,rgyy,rgyz],
               [rgxz,rgyz,rgzz]]
        
        rgM = np.square(rgM)
        
        eigV,eigM1 = np.linalg.eig(rgM)
        eigV = np.linalg.eigvals(rgM)
        
        eigM[index-1] = eigV
        
        shape = 'globular'
        if np.max(eigV)>20*np.min(eigV):
            shape = 'filamentous'
        elif np.max(eigV)>12*np.min(eigV):
            shape = 'semi-filamentous'
        
        # print(index,shape,'\n',eigV)
        
        values = [
            index,
            timestep,
            shape,
            x,
            y,
            z,
            int(len(cluster_sub_df)/3),
            std_xyz[0],
            std_xyz[1],
            std_xyz[2],
            np.sum(rg_xyz),
            rg_xyz[0],
            rg_xyz[1],
            rg_xyz[2],
            rg_avg,
            ]
        
        liste.append(values)
        
    col_names = ['id',
                 'timestep',
                 'conformation',
                 'x',
                 'y',
                 'z',
                 'size', 
                 'std_x',
                 'std_y',
                 'std_z',
                 'Rg',
                 'Rgx',
                 'Rgy',
                 'Rgz',
                 'Rg_avg']
    
    cluster_table = pd.DataFrame(data=liste,columns = col_names)
    if simple:
        return cluster_table[['timestep','conformation','size']]
    else:
        return cluster_table

def labeledGraph(arr_name:str,index:int=-1,
                 show_index:bool=False,savepng:bool=False)->plt.Axes:
    """
    Generate a labeled cluster graph.
    Label consist of clusterID and shape.

    Parameters
    ----------
    arr_name : str
        File to read.
    index : int, optional
        Time point. The default is -1.

    Returns
    -------
    ax : plt.Axes
        Labeled graph.

    """
    frame = np.load(arr_name)[index]
    ss = cluster_single_frame(frame)
    ax = imagify(ss)
    table = generate_cluster_info_table(frame,simple=False)
    
    for i in table.index:
        if show_index:
            label = f'{i+1}{table.conformation[i][:1].upper()}'
        else:
            label = f'{table.conformation[i][:1].upper()}'
            
        plt.text(x=table.x[i]+3, y=table.y[i]+3,
                 s=label,backgroundcolor='red',color='white',
                 fontweight='bold')
    
    plt.savefig(f'{arr_name[:-4]}.png',dpi=400)
    
    return ax


def makecircle(r:float=1, rotate:float=0, step:int=100)->pd.DataFrame:
    """

    Parameters
    ----------
    r : float, optional
        Radius of the circle. The default is 1.
    rotate : float, optional
        Radian angle to rotate the circle. The default is 0.
    step : int, optional
        Number of point in the circle. The default is 100.

    Returns
    -------
    coor : pd.DataFrame
        DataFrame with XYZ coordinates of the circle.

    """
    
    incre = (2*np.pi/step) #incrementation for desired steps
    angles =  np.arange(-np.pi,np.pi,incre)#len(angles) == step
    #initializing the array
    x = np.zeros(len(angles))
    y = np.zeros(len(angles))
    z = np.zeros(len(angles))
    
    #for every angle a point (or particle) is created
    for i,a in enumerate(angles):
        x[i] = r*np.cos(a)*np.sin(rotate)
        y[i] = r*np.sin(a)*np.sin(rotate)
        z[i] = r*np.cos(rotate)
        
    coor = pd.DataFrame()#empty DataFrame
    #coordinates are assigned as columns
    coor['x'] = x
    coor['y'] = y
    coor['z'] = z
    
    return coor

def makesphere(r:float,step:int=100)->pd.DataFrame:
    """

    Parameters
    ----------
    r : float, optional
        Radius of the circles -> sphere. The default is 1.

    step : int, optional
        Number of circles in the sphere. The default is 100.

    Returns
    -------
    coor : pd.DataFrame
        DataFrame with XYZ coordinates of the sphere.

    """
    incre = (2*np.pi/step) #incrementation for desired steps
    coor = makecircle(r,rotate=np.pi,step=step)
    
    circle_size = len(coor)#reference size of circle
    angles =  np.arange(-np.pi,np.pi,incre)
    sphere_size = circle_size*len(angles)
    
    sphere = np.zeros([sphere_size,3])#sphere array allocated
    
    #for every angle a circle is created
    for i,a in enumerate(angles):
        values = makecircle(r,rotate=a,step=step)
        sphere[i*circle_size:(i+1)*circle_size] = np.array(values) 
    
    coor = pd.DataFrame(sphere,columns= ['x','y','z'])#DataFrame formed
    
    return coor


def marksurface(arr:np.ndarray)->pd.DataFrame:
    """
    Marks surface atoms of XYZ np.ndarray into a dataframe.

    Parameters
    ----------
    arr : np.ndarray
        XYZ coordinates of the particles.

    Returns
    -------
    blob : pd.DataFrame
        DataFrame with particles marked as 0 for core 1 for surface in as an
        additional column.

    """
    
    r_max = np.ceil(np.sqrt(np.max(np.sum(arr**2,axis=1)))) # max of r_sqr is sqrt'ed
    size = len(arr)
    size_sqrt = np.sqrt(size)
    
    step = np.ceil(size_sqrt)*4
    if r_max>size_sqrt:
        step = np.ceil(np.sqrt(size))*5
        
    
    sphere = np.array(makesphere(r_max,step=step))
    # print(len(sphere))
    # print(f'{len(sphere):.1f}',end='\t')
    blob = pd.DataFrame(arr,columns= ['x','y','z'])
    blo_type = np.zeros(len(arr))#type is surface or not surface (1 or 0)
    closest = np.zeros(len(sphere))#to store closest partcl. for each sp. point 
    
    for i,s in enumerate(sphere):
        dist_sqr = np.sum((arr-s)**2,axis=1)#xyz(all)-xyz[i]
        # dist = np.sqrt(dist_sqr)
        index_min = np.argmin(dist_sqr)
        closest[i] = index_min
        
    unique_close = np.unique(closest)
    blo_type[unique_close.astype(int)] = 1
    blob['surface'] = blo_type
    
    return blob

def mark_cluster_surface(frame:pd.DataFrame)-> pd.DataFrame:
    """
    Identifies surface atoms of cluster, 
    marks them with 1 for surface 0 otherwisse
    

    Parameters
    ----------
    frame : pd.DataFrame
        single frame from dump array.

    Returns
    -------
    cluster_frame : pd.DataFrame
        frame with new column of 'surface' marked 1 if surface else 0.

    """
    
    
    cluster_frame = frame[frame.clusterID>0]#remove non-cluster particles

    
    
    cluster_frame['id'] = np.arange(len(cluster_frame))#giving ids to atoms
    num_clusters = cluster_frame.clusterID.max()
    
    #initing array to store surface info
    surface = np.zeros([len(cluster_frame)])
    
    for i in range(int(num_clusters)):
        #subslicing of clusters and only hinge domain of the protein
        cl = cluster_frame[(cluster_frame.clusterID==i+1) & (cluster_frame.type==5)]
        ids = np.array(cl['id'])
        xyz = np.array(cl[['x','y','z']])
        
        #normalization of the current cluster coordinates
        avg_xyz = np.average(xyz,axis=0)
        xyz_norm = xyz-avg_xyz
        
        #marking surface atoms
        cl = marksurface(xyz_norm)
        surface[ids] = cl.surface
        # cluster_frame.surface.iloc[ids] = cl.surface

    cluster_frame['surface'] = surface
    
    return cluster_frame

def system_cluster_summary(file:os.PathLike,num_samples:int=20)->pd.DataFrame:
    """
    

    Parameters
    ----------
    file : os.PathLike
        path/to/file.
    num_samples : int, optional
        Number of timestep to take sample. The default is 20.

    Returns
    -------
    full_sumarry : pd.DataFrame
        Dataframe containing cluster info about all timestesp

    """
    dump = np.load(file)
    
    #length arrangment and selection of random time points
    timestep = len(dump)
    ignore = int(len(dump)/20)
    timepoints = np.random.randint(low=ignore,high=timestep-ignore,size=num_samples)

    full_summary = pd.DataFrame()
    for timestep in track(timepoints):
        
        info_table = generate_cluster_info_table(dump[timestep],timestep=timestep)
        full_summary = pd.concat([full_summary,info_table])
    
    return full_summary

def Rstyle(func:callable,**kwargs)->plt.Axes:
    
    font = {'family': 'sans-serif',
            'weight': 'light',
            'size': 14,
            }
    sns.set(style='ticks',
        rc = {'figure.figsize':(5.6,4.2),
              'font.weight':'light',
              'font.family':'sans-serif',
              'axes.spines.top':'False',
              'axes.spines.right':'False',
              'ytick.minor.size':'0',
              'ytick.major.size':'10',
              'xtick.major.size':'10',
              'legend.frameon':False
              
              }
        )
    
    props = {
    'boxprops':{'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
    
    }
    
    ax = func(**kwargs,**props)
    
    lower_auto,upper_auto = plt.gca().get_ylim()
    
    if kwargs.get('yscale') == 'log':
        lower = 0.1
        exp = int(np.log10(upper_auto))
        upper = 10**(exp+1)
    
    else:
        lower=0
        exp = 10**int(np.log10(upper_auto))
        upper = (int(upper_auto/exp)+1)*exp
        
    plt.ylim([lower,upper])

    return ax


if __name__ == '__main__':
    #changing working directory to current directory name
    os.chdir(os.path.dirname(__file__))

    
    # df = system_cluster_summary('./dump.npy',num_samples=20)
    df = system_cluster_summary('./sampleDumps/dump.npy',num_samples=20)
    df.to_csv('clusterSummary.csv',index=False)
    
    ax = Rstyle(sns.violinplot,data=df,
                x='conformation',y='size',
                palette='bright',saturation=1,linewidth=1)
    
    plt.savefig('clusterSummary.png',dpi=400)
    
    


