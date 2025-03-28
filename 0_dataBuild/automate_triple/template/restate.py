#dump2arr
import numpy as np
import time

def txt2arr(ftoread="dump.main",ignore = 1001):
    """transferring dump data into two 3D array
    Polymer beads and free molecules are transferred to different arrays.
    """
    
    tos = time.perf_counter() #start time

    ignore_num = ignore
    #reading dump file line-by-line--------------------------------------------
    dump = open(ftoread, "r")
    lines = dump.readlines()
    dump.close()
    
    n = 0#dna monomer
    freetf = 0#free tf
    bs = 0#biding site
    
    #finding the number of the atoms for each type and number of the lines between timesteps-----
    
    number_atom = int(lines[3])
    batch = number_atom + 9
    
    for a in lines[9:batch+1]:
        b = a.split()[1]
        if b == "1" :
            n +=1
        if b == "2":
            n +=1
            bs +=1
        elif b == "3":
            freetf+=0.5
    
    freetf = int(freetf)
    bs = int(bs/3)
    
    TF = freetf #number of tfs
    
    print(n, "DNA monomers",end=" ")
    print(freetf, "Free Transcriptions factors",end=" ")
    print(int(n*10/(TF)), "Bp per TF")
    
    #number of the timesteps---------------------------------------------------
    time_num = int(len(lines)/batch)

    #creating an array of zeros with intented size-----------------------
    time_interval = time_num-ignore_num
    dna =  np.zeros([time_interval,n,3],dtype=np.float32)
    naps = np.zeros([time_interval,TF*2,3],dtype=np.float32)
    #data = np.zeros([time_num,number_atom,3])
    
    #for loop to create array from the dump file-------------------------------
    for i in range(ignore_num,time_num):
        #for loop to make the 3D array-----------------------------------------
        

        for k in range(number_atom):

            values = lines[i*batch+k+9].split()
            atomID = int(values[0])
            
            #polymer atoms
            if atomID<=n:
                dna[i-ignore,atomID-1,0] = float(values[2])
                dna[i-ignore,atomID-1,1] = float(values[3])
                dna[i-ignore,atomID-1,2] = float(values[4])
            
            #free atoms
            else:
                index = atomID-1-n
               
                #condition exclude nonbinding domains       
                if atomID %3 !=2:
                    rem = int((index+1)/3)
                    naps[i-ignore,index-rem,0] = float(values[2])
                    naps[i-ignore,index-rem,1] = float(values[3])
                    naps[i-ignore,index-rem,2] = float(values[4])
        
        tof = time.perf_counter() #finish time
        if i%(int(time_interval/100))==0 or i == time_num-1:
            progressbar(i-ignore_num+1, time_interval, tof-tos, prefix = "Text to array")
        
    tfd = int(n/bs)    
    

    print(f'{tof-tos:.1f} Secods to dump 2 array conversion...')
    return dna,naps,tfd

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

            if np.any(np.less(distance,1)): # if less than threshold distance
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

if __name__ == '__main__':

    dna,naps,tfd = txt2arr()
   
    l_dna = len(dna[0])
    sites = np.arange(l_dna)
    sp_sites = [] #specific sites
    ns_sites = [] #nonspecific sites
    for i in range (len(dna[0])):
        if i%tfd == 0 or i%tfd == 1 or i%tfd == 2:
            sp_sites.append(i)
        else:
            ns_sites.append(i)
            
    sp_dna = dna[:,sp_sites].copy() # only sp sites
    ns_dna = dna[:,ns_sites].copy() # only ns sites
    
    #comparisons for both legs       
    
    sp_state = restate3D(sp_dna, naps, work_title='SP  ') #sp and naps
    ns_state = restate3D(ns_dna, naps, work_title='NS  ') #ns and naps
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

    np.savez('states',nap_state=nap_state,ns_state=ns_state,sp_state=sp_state,
                    lfoot=lfoot,rfoot=rfoot,
                    sp_lfoot=sp_lfoot,sp_rfoot=sp_rfoot,
                    ns_lfoot=ns_lfoot,ns_rfoot=ns_rfoot,
                    fl_nap=fl_nap,sp_nap=sp_nap,ns_nap=ns_nap,
                    fl_lfoot=fl_lfoot,fl_rfoot=fl_rfoot,fl_feet=fl_feet)