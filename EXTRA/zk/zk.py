import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def find_cluster(atoms,threshold=2.1,min_size=5):
    """Takes an array(atoms) with the coordinates of atoms
    clusters atoms that are close to each other less
    than threshold.

    Returns cluster_arr 
    row indexes: unique clusters
    column indexes: atoms, marked 1 if associated.

    """
    num = len(atoms)
    groups = np.zeros((num,num),dtype='float32')
    
    #neighbors lists for each atom is found
    for i in range(num):
        distance = np.linalg.norm(np.subtract(atoms,atoms[i]),axis=1)#euclidian distance of distances in each (x,y,z) direction
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

def data2df(file):
    """ Turns LAMMPS data file into DataFrame 
    via retrieving atoms and their information.
    with columns:'atomID','type','xco','yco','zco'
    """
    
    f = open(file,'r')
    
    lines = f.readlines()
    start,end = 0,0
    
    #finding beginning and the end of the atoms
    for i,l in enumerate(lines):
        if l[0:5] == 'Atoms':
            start = i+2
        if l[0:10] == 'Velocities':
            end = i-1
            break
    
    atom_lines = lines[start:end]
    
    coor = [a.split(' ') for a in atom_lines] #string to list
    coor = [c for c in coor if c != ['\n']] #remove empty lines if present
    coor = [c for c in coor if c[1] != '6'] #remove capsid
    
    rows = len(coor)
    cols = 5
    col = [0,1,3,4,5]#
    
    
    atoms = np.zeros([rows,cols])
    
    for c in coor:
        for i,j in enumerate(col):
            atomID = int(c[0])-1
            atoms[atomID,i] = float(c[j])
            
    
    names = ['atomID','type','xco','yco','zco']
    df_atoms = pd.DataFrame(atoms, columns = names)
    return df_atoms

def cluster_id(cluster_arr,n=0):
    """
    from cluster_arr of clusters and atoms marked to be contained,
    returns a list (numpy array) that marks atoms with their respective
    cluster ids. 

    if there are not-to-be-clustered atoms with number (n) they are 
    initiated as 0 before the cluster atoms.
    """

    width = len(cluster_arr[0])+n #cluster atoms + other atoms
    cluster = np.zeros([width])

    for j,c in enumerate(cluster_arr):
        cluster_num = j+1 #giving ids to the cluster
        cluster_atoms =  np.where(c==1)[0] #indexes of atoms
        index_in_df = n+cluster_atoms  #indexes in the df (n, number of not-to-be-clustered atoms)
        cluster[index_in_df] = cluster_num

    return cluster

def draw_cluster(fname):
    """
    Draw cluster from the given file
    """

    #file and graph labels
    numbers = [x for x in fname if x.isdigit()]
    kt_name = "".join(numbers)
    kt_val = float(kt_name)/10

    df_atoms = data2df(fname)
    #proteins
    dd = df_atoms[df_atoms['type']>2]
    #acquring dna length
    dna = df_atoms[df_atoms['type']<3]
    n = len(dna)
    
    atom_ids = np.array(dd['atomID'])
    #simplifying df to coordinates array
    d = dd[['xco','yco','zco']]
    d = np.array(d)
    #cluster arrays
    cl = find_cluster(d)
    cluster_size = np.sum(cl,axis=1)#sizes of each array
    
    #adding clusters to df
    df_atoms['cluster'] = cluster_id(cl,n)
    #df
    df_clusters = df_atoms[df_atoms['cluster']>0]
    
    #figure plotting
    sns.set(rc = {'figure.figsize':(14,6)})
    sns.set_style(style = 'white')
    sns.scatterplot(data=df_clusters, 
                    x = 'xco',y = 'yco',
                    hue='cluster',
                    palette='bright'
                    )
    plt.grid(False)
    plt.xlim([-70,70])
    plt.ylim([-30,30])
    plt.title(str(kt_val)+'kT')
    plt.savefig(kt_name+'_cluster.png',dpi = 300)
    plt.figure()

    return

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
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'],axis=1)
    for i,col in enumerate(df.columns): #first is the time column
        df.iloc[:,i] = arr_minimize(df.iloc[:,i]) #values minimized and returned


    return df
