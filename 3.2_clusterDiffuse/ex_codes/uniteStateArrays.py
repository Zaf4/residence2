import numpy as np
import os


def generateLocations(filename:str,
                      main_loc:str = '/home/gottar/5x10t'):
    
    locations = []
    
    cases = ['/1.00','/2.80','/3.00','/3.50','/4.00']
    concs = ['/10','/20','/40','/60']
    
    for case in cases:
        for conc in concs:
            locations.append(main_loc+case+conc+filename)
            
    return locations

def unite_states(file:os.PathLike,
                 savename:os.PathLike)->np.ndarray:
    
    ns = np.load(file)['ns_state']
    sp = np.load(file)['sp_state']
    
    fl = ns+sp
    fl[fl==2]=1
    
    fl = fl.astype(bool)
    np.save(savename,fl)
    
    return 

if __name__ == '__main__':
    #locations and filesavenames
    locations = generateLocations('/states.npz')
    savenames = generateLocations('/full.npy')
    
    for file,savename in zip(locations,savenames):
        unite_states(file, savename)
    
    