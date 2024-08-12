import numpy as np
import os
import sys

def read_residence(resident):
    
    def resider(val=1):
        """"
        Measures the time of staying bound.
        Appends them to a list and return the list.
        """
        res_time = []
        
        for res in resident: #for each row or the binding molecule
            time = 0 #set occurence time to zero
            for r in res: #binding status 0 or 1,2
                if r:
                    time+=1
                elif r==0: #when the binding status is not satisfied anymore, binding duration is saved and time reset to 0
                    res_time.append(time)
                    time = 0
        return np.array(res_time)   
             
    # unbound = resider(0) #unbound residences
    bound = resider(1) #bound residences
    
    val_max = len(state[0]) #max time value to decide array length
    length = val_max
    serial = np.arange(1,val_max+1,dtype=int) #time values 
    distribution = np.zeros([length,2],dtype='float32')#empty array to store durations vs occurence rates
    distribution[:,0] = serial #first row is durations
    
    for i in range(length):#counting occurences
    
        distribution[i,1] = np.count_nonzero(bound == distribution[i,0]) #how many occurences for occurences observed for each duration

    
    return distribution[:,1].copy()


# def savingloc():
    

if __name__ == '__main__':
    state = np.load('states.npz')['fl_nap']
    dist = read_residence(state)
    dist = np.where(dist == 0, np.nan, dist) #zeros removed
    
    #determining the name of the file and saving location
    sepr = "/" #for mac and linux
    if sys.platform == 'win32':
        sepr = '\\'
    
    cloc = os.getcwd().split(sepr)
    
    arr_name = cloc[-3]+'_'+cloc[-2]+'_'+cloc[-1]+'.npy'
    save_l = cloc[:-2]
    
    save_loc = sepr.join(save_l)
    
    os.chdir(save_loc)
    
    np.save(arr_name, dist)
    print(f'{arr_name} saved to {save_loc}')