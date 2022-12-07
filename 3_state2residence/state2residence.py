import numpy as np
import pandas as pd
import os
from time import perf_counter as pf

def state2residence(zipname:str='states.npz')->np.ndarray:
    state_ns = np.load(zipname)['sp_state.npy'].astype(int)
    state_sp = np.load(zipname)['ns_state.npy'].astype(int)
    state = state_ns+state_sp
    state[state>1] = 1
    
    #rows->num(tf) and cols->num(timestep)
    num_tf,num_time = state.shape
    
    #empty array for storing durations
    durations = np.zeros(num_time).astype(int) #time values to be stored
    
    lifetimes = np.sum(state,axis=1) #also to be used later
    #to be ignored row -- if all bound or all unbound,
    skips, = np.where((lifetimes==0) | (lifetimes==num_time))
    maxes, = np.where(lifetimes==num_time)#dyration = tmax
    durations[-1] = len(maxes)
    
    #for each row(protein)
    for i,st in enumerate(state):
        # print(f'{i+1}/{num_tf}   ',end='\r',flush=True)
        if i in skips:
            continue
        dur = 0
        #for each time step
        for s in st:
            if s:
                dur += 1
            elif dur>0 and s==0:
                durations[dur-1]+=1
                dur=0#durations is reset
    

    return durations


def locations():
    
    home = '/home/gottar'
    init = home+'/5x10t'
    save_dir = init+'/csv'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cases = ['/1.00','/2.80','/3.00','/3.50','/4.00']
    concs = ['/10','/20','/40','/60']

    locs,save_names = [],[]
    for ca in cases:
        for co in concs:
            locname =init+ca+co
            savename = ca[1:]+'_'+co[1:]
            locs.append(locname) #locations list
            save_names.append(savename)#saving names list
            
    return locs,save_dir,save_names


if __name__ == '__main__':
    
    locs,save_dir,save_names = locations()
    df = pd.DataFrame()
    size = len(state2residence(locs[0]+'/states.npz'))
    for loc,name in zip(locs,save_names):
        start = pf()
        name.replace('.','')
        arr = state2residence(loc+'/states.npz')
        if len(arr) != size: #size fixing
            arr_fixed = np.zeros(size)
            arr_fixed[:len(arr)] = arr
            df[name] = arr_fixed
        
        else:
            df[name] = arr
        
        df.to_csv(save_dir+'/durations.csv')
        print(f'{name}\t{pf()-start:.1f}s')
