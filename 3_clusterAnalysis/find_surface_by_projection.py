import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def add_index(arr):
    #adding indexes
    arr_wi = np.zeros([len(arr),4])
    arr_wi[:,1:] = arr
    arr_wi[:,0] = np.arange(len(arr))
    
    return arr_wi

def find_surface_2d(arr:np.ndarray,projection:np.ndarray,npa:int)->np.ndarray:
    """

    Parameters
    ----------
    arr : np.ndarray
        cluster array.
    projection : np.ndarray
        its projection.
    npa : int
        non-projection axis- for xy projection z is npa.

    Returns
    -------
    closest_points : np.ndarray
        closest point for the given projection.

    """
    #adding indexes

    
    closest_points = np.zeros(len(arr))
    for i,prj in enumerate(projection):
        co = [1,2,3]
        co.remove(npa)
        distance2d = np.sum(np.square(arr[:,co]-prj[co]),axis=1)#distance in 2d
        partial = arr[distance2d<0.2]#slice of arr that is less than 1 away in 2d
        axial_dist = partial[:,npa]-prj[npa]#distance in non-projection axis
        index_in_axial = np.argmin(axial_dist**2)
        closest_points[i] = partial[index_in_axial,0]
        
    # unique_closest = np.unique(closest_points).astype('int16')


    return closest_points



def project2d(arr:np.ndarray)->list:
    """

    Parameters
    ----------
    arr : np.ndarray
        xyz coordinates of particles.

    Returns
    -------
    list
        list of 6 2d-projection.

    """
    # arr = add_index(arr)
    
    xy_low,xy_high,xz_low = arr.copy(),arr.copy(),arr.copy()
    xz_high,yz_low,yz_high = arr.copy(),arr.copy(),arr.copy()
    
    xi,yi,zi = 1,2,3
    #xy
    xy_low[:,zi] = np.min(arr[:,zi])
    xy_high[:,zi] = np.max(arr[:,zi])
    #xz
    xz_low[:,yi] = np.min(arr[:,yi])
    xz_high[:,yi] = np.max(arr[:,yi])
    #yz
    yz_low[:,xi] = np.min(arr[:,xi])
    yz_high[:,xi] = np.max(arr[:,xi])
    
    return [xy_low, xy_high, xz_low, xz_high, yz_low, yz_high]


def marksurface_projection(arr:np.ndarray)->pd.DataFrame:
    arr = add_index(arr)
    sides = project2d(arr)
    npas = [3,3,2,2,1,1]
    
    #finds all particles that are surface non-unique fashion
    surface_index = np.zeros([6,len(arr)])
    for i,npa,side in zip(range(6),npas,sides):
        surface_index[i] = find_surface_2d(arr, projection=side, npa=npa)
    
    #remove the repeats 
    unique_closest = np.unique(surface_index).astype('int')
    
    #initializing and marking the surface particles
    surf = np.zeros(len(arr))
    surf[unique_closest] = 1       
         
    #creating the relevant DataFrame
    blob = pd.DataFrame(arr[:,1:],columns=['x','y','z'])
    blob['surface'] = surf
    
    return blob
    
arr = np.random.rand(12,3)*6-3
blob = marksurface_projection(arr)
core = blob[blob.surface==0]
surf = blob[blob.surface==1]


fig = plt.figure(figsize=(9,9))
ax = plt.axes(projection='3d')
ax.scatter(core.x,core.y,core.z,
          marker='o',alpha=0.4,
          color='grey',s=40)
ax.scatter(surf.x,surf.y,surf.z,
          marker='o',alpha=0.6,
          color='green',s=40)
plt.show()

sns.scatterplot(data=blob,x='x',y='y',hue='surface',size='z',palette='bright')
