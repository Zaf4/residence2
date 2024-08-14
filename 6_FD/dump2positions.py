import numpy as np
import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("-k", "--kt", type=str, help="kinetic temperature")
args = argparser.parse_args()

kt = args.kt
 
def is_bound(arr_DNA:np.ndarray,tf:np.ndarray,threshold:float=0.8) -> np.ndarray:

    # tf is a single row 
    a = np.subtract(arr_DNA,tf) #distance in each (x,y,z) direction
    distance = np.linalg.norm(a,axis=1)#euclidian distance

    if np.any(np.less(distance,threshold)): # if less than threshold distance
        return True
    else:
        return False

 
def where_bound(arr_DNA:np.ndarray,tf:np.ndarray,threshold:float=0.8)->np.ndarray:

    """
    Return the index of the DNA atom where the given tf is bound.
    if tf is not bound to DNA at all, returns np.nan

    """
    n_dna = len(arr_DNA)
    index = np.arange(n_dna).astype(np.uint32)
    # tf is a single row 
    a = np.subtract(arr_DNA,tf) #distance in each (x,y,z) direction
    distance = np.linalg.norm(a,axis=1)#euclidian distance

    if np.any(np.less(distance,threshold)):
        return index[np.less(distance,threshold)][0] # only single one
    else:
        return np.nan

 
def find_positions(arr_DNA:np.ndarray,
                   arr_tf:np.ndarray)->np.ndarray:
    
    """
    returns all tfs' position on DNA at a single timestep

    Returns
    -------
    nd.array
        1d array of length = len(arr_tf)
        positions of each tf on DNA polymer
    """
    
    positions = np.zeros(len(arr_tf)) 
    for i, tf in enumerate(arr_tf):
        where = where_bound(arr_DNA,tf,threshold=0.9)
        positions[i] = where

    return positions

 
def find_positions_multistep(arr_DNA_ms:np.ndarray,arr_tf_ms:np.ndarray)->np.ndarray:

    """
    returns all tfs' position on DNA at all timesteps

    Returns
    -------
    nd.array
        2d array of nrow = n_tf and ncol = n_timestep
        positions of each tf on DNA polymer at all timestep
    """

    n_timestep, n_tf, n_dim = arr_tf_ms.shape

    all_positions = np.zeros([n_tf,n_timestep]) # each col will be a timestep for rows (tfs)

    for i in range(n_timestep):
        all_positions[:,i] = find_positions(arr_DNA_ms[i],arr_tf_ms[i])
        print(f"step: {i+1} out of {n_timestep}", end='\r')
    print()
    return all_positions


def save_positions(fname:str="dump.npy")->None:

    arr = np.load(fname) #timestep 1001 to 2001
    n_time, n_atoms, n_dim = arr.shape

    atom_types = arr[0,:,0]
    atom_types
    atom_id = np.arange(n_atoms)+1
    n_DNA = sum((atom_types == 2) | (atom_types == 1))# number of DNA beads
    n_tf = sum(atom_types == 5)

    # conditions 
    condition_L = ((atom_types == 3) & (atom_id % 3 == 1))
    condition_H = ((atom_types == 5) & (atom_id % 3 == 2)) # second statement in unnecessary
    condition_R = ((atom_types == 3) & (atom_id % 3 == 0))

    # array gruops 
    polymer_DNA = arr[:,:n_DNA,1:]
    left_legs = arr[:,condition_L,1:]
    right_legs = arr[:,condition_R,1:]
    
    # the positions of each tf for each time step per leg
    print("L")
    positions_L = find_positions_multistep(polymer_DNA,left_legs)
    print("R")
    positions_R = find_positions_multistep(polymer_DNA,right_legs)

    np.savez("positions.npz", positions_L,positions_R)

    return

def main(kt:str):

    UMS = ["10", "20", "40", "60"]

    for um in UMS:
        folder = f"~/5x10t/{kt}/{um}"
    
    os.chdir(folder)
    save_positions("dump.npy")

if __name__ == '__main__':
    main(kt=kt)

 



