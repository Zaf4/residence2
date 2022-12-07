import numpy as np
import os
import pandas as pd

def data2df(file:os.PathLike):
    """ Turns LAMMPS data file into DataFrame 
    via retrieving atoms and their information.
    with columns:'atomID','type','xco','yco','zco'
    """
    
    f = open(file,'r')
    
    lines = f.readlines()
    f.close()
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