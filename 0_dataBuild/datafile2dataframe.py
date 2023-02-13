import numpy as np
import os
import pandas as pd

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
    names = ['atomID','type','xco','yco','zco']
    df_atoms = pd.DataFrame(atoms, columns = names)
    
    return df_atoms

if __name__ == '__main__':
    df = data2df('./data.denge')
    tfs = df[df.type==3]
    indexes = tfs.index
    tfarr = np.array(tfs.type).astype(np.int32)
    size = len(tfs)
    
    perc8 = int(size*0.08)
    perc12 = int(size*0.12)
    perc16 = int(size*0.16)
    
    #creating random arrays to modify tf types in accord with gaussian dist.
    low0 = np.random.randint(0,size,size=perc8)
    low1= np.random.randint(0,size,size=perc12)
    low2 = np.random.randint(0,size,size=perc16)

    high0 = np.random.randint(0,size,size=perc16)
    high1 = np.random.randint(0,size,size=perc12)
    high2 = np.random.randint(0,size,size=perc8)
    
    #assinging types to index
    #16 percents
    tfarr[low2] = 9
    tfarr[high0] = 11
    #12 percents
    tfarr[low1] = 8
    tfarr[high1] = 12
    #8 percents
    tfarr[high2] = 13
    tfarr[low0] = 7
    
    #changing it in dataframe
    arr = np.array(df.type)
    arr[indexes] = tfarr
    df.type = arr
        
    
    
    
    
    
    
    
    # arr = np.array(df)