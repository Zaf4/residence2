import time
import numpy as np
import os


def txt2arr(ftoread: str = "dump.main",ignore: int = 0)->np.ndarray:
    """
    transferx dump data into two 3D numpy array.
    Only the types and coordinates preserved.
    """
    
    tos = time.perf_counter() #start time

    ignore_num = ignore
    #reading dump file line-by-line--------------------------------------------
    dump = open(ftoread, "r")
    size = os.stat(ftoread).st_size
    lines = dump.readlines()
    dump.close()
    n = 0#dna monomer

    #finding dna polymer count
    number_atom = int(lines[3])
    batch = number_atom + 9
    
    for a in lines[9:batch+1]:
        b = a.split()[1]
        if b == "1" :
            n +=1
    
    #number of the timesteps---------------------------------------------------
    time_num = int(len(lines)/batch)
    print(time_num)
    #creating an array of zeros with intented size-----------------------
    time_interval = time_num-ignore_num
    arr =  np.zeros([time_interval,number_atom,4],dtype=np.float32)
    
    #for loop to create array from the dump file-------------------------------
    for i in range(ignore_num,time_num):
        
        #for loop to make the 3D array-----------------------------------------
        for k in range(number_atom):

            values = lines[i*batch+k+9].split()
            atomID = int(values[0])
            
            #polymer atoms
            if True:
                arr[i-ignore,atomID-1,0] = float(values[1])
                arr[i-ignore,atomID-1,1] = float(values[2])
                arr[i-ignore,atomID-1,2] = float(values[3])
                arr[i-ignore,atomID-1,3] = float(values[4])
            
        tof = time.perf_counter() #finish time
        if i%(int(time_interval/100))==0 or i == time_num-1:
            progressbar(i-ignore_num+1, time_interval, tof-tos, prefix = "Text to array")
        
    with open('summary.txt','w') as summary:
        summary.write(f'DNA monomers: \t\t{n}\n')
        summary.write(f'TF atoms: \t\t\t{number_atom-n}\n')
        summary.write(f'TF molecules: \t\t{(number_atom-n)/3}\n')
        summary.write(f'bp/TF: \t\t\t{n*10/((number_atom-n)/3):.2f}\n')
        summary.write(f'Dump file size: \t\t{size/(2**30):.2f} GB\n')
        summary.write(f'Numpy Array Size: \t{arr.size*arr.itemsize/2**30:.2f} GB\n')
        summary.write(f'Compression level: \t{100-arr.size*arr.itemsize/size*100:.2f}%\n')
        summary.write(f'Conversion Time: \t\t{tof-tos:.1f} seconds\n')
        
    
    print(f'{tof-tos:.1f} Seconds to dump 2 array conversion...')
    return arr

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
    arr = txt2arr()
    np.save('dump.npy',arr)
    arr_small = arr[-1000:]
    np.save('small.npy',arr_small)