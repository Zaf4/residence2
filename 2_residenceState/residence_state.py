#dump2arr
import numpy as np
import time

def restate2D(arr1,arr2):
    """ Creates a zeros array by the size of the beads.
    Finds distances between the polymer and the free beads. 
    If a bead is near a polymer, it is considered bound (1) in the state array.
    Returns the state array.
    """
    lenf = len(arr2) #molecule to compare
    state= np.zeros(lenf,dtype=np.float32) #bound or not

    for i in range(lenf): #compare every free bead
            a = np.subtract(arr1,arr2[i]) #distance in each (x,y,z) direction
            distance = np.linalg.norm(a,axis=1)#euclidian distance

            if np.any(np.less(distance,0.9)): # if less than threshold distance
                state[i] = 1  #marked as bound

    return state


def restate3D(arrPol,arrMol, work_title = "Computing"):
    """
    Uses restate2D for every time depth.
    """
    time_range = len(arrPol) 
    mol_num = len(arrMol[0])
    
    state_whole = np.zeros([time_range,mol_num],dtype=np.float32) #creates a 3D state array to contain state for each bead at each time
    time_earlier = time.perf_counter()
    for i in range(time_range):
        state_whole[i] = restate2D(arrPol[i], arrMol[i])
        
        #performs occasional timing to estimate ending time.
        if i%10 == 1 or i == time_range-1:
            end_time = time.perf_counter()
            time_diff = end_time-time_earlier
            progressbar(i+1, time_range, time_diff,prefix=work_title)
    
    print(f'{work_title} done in {time_diff/60:.1f} minutes')
    
    return state_whole.T #formatting for the following code

def progressbar(step, length, time_elapsed,prefix = "Computing"):

    per = step/time_elapsed #performance
    remaining = length-step	#remaining steps

    #max bar length
    max_bar = 40
    step_norm = int(step/length*max_bar)
    remaining_norm = max_bar-step_norm

    eta = remaining/per

    unit = 's'
    if eta>60 and eta<3600:
        eta = eta/60
        unit = 'min'
    elif eta>3600:
        eta = eta/3600
        unit = 'h'

    ending = '\r'
    if step == length:
        ending = '\n'

    print(f'{prefix}: [{u"█"*step_norm}{" "*remaining_norm}]{step}/{length} -- Step/s:{per:.1f} -- ETA:{eta:.1f}{unit}     ',flush=True,end=ending)

def state_all(fname:str='dump.npy',ignore:int=1001)->None:
    """
    Creates a '.npz'file (using the dump file)
    storing all the state arrays and saves it.

    Parameters
    ----------
    fname : str, optional
        dump file to be read. The default is 'dump.npy'.
    ignore : int, optional
        DESCRIPTION. The default is 1001.

    Returns
    -------
    None
        DESCRIPTION.

    """
    arr = np.load(fname)
    part = arr[0]
    #slicing part to identify the indexes of dna sites and naps
    # dna_ind = np.where(part[:,0]<3)[0]
    sp_sites = np.where(part[:,0]==2)[0]
    ns_sites = np.where(part[:,0]==1)[0]
    print(len(sp_sites),len(ns_sites))
    naps = part[(part[:,0]>2)]
    nap_ind = np.where(naps[:,0]!=5)[0]+len(sp_sites)+len(ns_sites)
    
    arr = arr[ignore:,:,1:]
    
    naps = arr[:,nap_ind].copy()      
    sp_dna = arr[:,sp_sites].copy() # only sp sites
    ns_dna = arr[:,ns_sites].copy() # only ns sites
    arr = 0
    #comparisons for both legs       
    
    sp_state = restate3D(sp_dna, naps, work_title='SP  ') #sp and naps
    ns_state = restate3D(ns_dna, naps, work_title='NS  ') #ns and naps
    """
    nap_state = np.add(ns_state,sp_state)
    nap_state[nap_state==2] = 1
    
    #legs from whole
    lfoot = nap_state[0::2].copy() #leg1s
    rfoot = nap_state[1::2].copy() #leg2s
    #sp ---legs
    sp_lfoot = sp_state[0::2].copy() #leg1s
    sp_rfoot = sp_state[1::2].copy() #leg2s
    #ns ---legs
    ns_lfoot = ns_state[0::2].copy() #leg1s
    ns_rfoot = ns_state[1::2].copy() #leg2s
     
    #addition of for full view
    fl_nap = np.add(lfoot,rfoot) #partials (foots) are added together 1 means partial 2 means complete binding
    sp_nap = np.add(sp_lfoot,sp_rfoot) #only sp interactions, 0:free, 1: one foot only, 2: both feet
    ns_nap = np.add(ns_lfoot,ns_rfoot) #only ns interactions, 0:free, 1: one foot only, 2: both feet
    
    #addition of sp to ns
    fl_lfoot = np.add(sp_lfoot,ns_lfoot) #left foot,- 0:free,1:only sp or ns, 2:both
    fl_rfoot = np.add(sp_rfoot,ns_rfoot) #right foot, 0:free,1:only sp or ns, 2:both
    #addition of both feet to view number of  interactions
    fl_feet = np.add(fl_rfoot,fl_lfoot) #gives how many sites a nap interacts
    """
    np.savez('states',ns_state=ns_state,sp_state=sp_state)

if __name__ == '__main__':
    state_all()

