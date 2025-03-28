
def steal(fname,tfd,wdn=1):
    """
    takes file name of lammps data file and steals their id and positions and 
    returns an array with all the atoms with their atom and molecule types 
    according to given tfd(for how many beads there is a binding site).
    """
    import numpy as np
    
    rr = open(fname,"r")
    
    #reading data file
    coor = rr.readlines()
    rr.close()
    atomNum = coor[2].split()
    atomN = atomNum[0]
    atmn = int(atomN)
    
    index = 0
    for rw in coor[0:100]:
        if rw != "\n":
            atm = rw.split()[0]
        #when Atoms string encountered it will be marked as start point
        if atm == "Atoms": 
            start = index+2
            break
        index +=1
        
    finish = start + int(atomN)
    
    ncol = 6
    
    #array of zeros formed with the neccesarry shapes
    atoms = np.zeros([atmn,ncol])
    
    #all the lines with atoms will be read
    for row in coor[start:finish]:
        atomd = row.split()[0]
        atomid = int(atomd)
        for j in range (ncol):
            atoms[atomid-1,j] = row.split()[j]
    
    #taking avarages
    ax ,ay , az = 0,0,0    
    for p in range(atmn):
        ax += atoms[p][3]/atmn
        ay += atoms[p][4]/atmn
        az += atoms[p][5]/atmn
        
    #centerilazing atoms
    for k in range(atmn):
        atoms[k][3] = atoms[k][3]-ax
        atoms[k][4] = atoms[k][4]-ay
        atoms[k][5] = atoms[k][5]-az
    
    #converting atom and molecule types to 1
    atoms[:,1:3]=1
    
    #converting atom and molecule types to 2 according to tf density
    for index,l in enumerate(atoms):
        if index%tfd in [0,1,2]: #zero is the center of binding sites; others side
           l[1:3] =2
           atoms[index]=l
    #atoms array is returned
    if wdn == 1:
        return atoms
    if wdn > 1:
        widened = widen(fname,tfd,wdn)
        return widened

"""_________________________________________________________________________"""
def widen(fname,tfd,wdn):
    """ 
    Creates widened data files using lammps data files, takes fname to use 
    steal method, and requires wdn, widening factor and tfd, transcription 
    factor density as args. 
    Returns widened array of atoms
    """
    
    import numpy as np
    
    atoms = steal(fname,tfd,1)
    watoms = atoms*(wdn**(2/3))
    boy = len(atoms)
    yeniboy = wdn*boy
    
    arr = np.zeros([yeniboy,6])
    
    a = 0
    for i,o in enumerate(arr):
        if (i+1)%wdn == 0 and i>0:
            arr[i]= watoms[a]
            a+=1
            
        
    widen = arr.copy()
    
    
    #fills between x+wdn and x+2wdn

    for i, row in enumerate(widen):
        if (i+1)%wdn == 0 and (i+1)%(2*wdn) != 0 and i > 0:
            coor1 = row [3:]

        elif (i+1)%wdn == 0 and (i+1)%(2*wdn) == 0 and i>0:
            coor2 = row [3:]
            dif = coor2-coor1
            difc = dif/wdn
        
            for p in range(1,wdn):
                widen[i-wdn+p][3:] = coor1+difc*p
                
    #fills between n+2wdn and n+2wdn   
    for i, row in enumerate(widen):
        if (i+1)%wdn == 0 and (i+1)%(2*wdn) == 0 and i > 0:
            coor1 = row [3:]
        elif (i+1)%wdn == 0 and (i+1)%(2*wdn) != 0 and i>0:
            coor2 = row [3:]
            dif = coor2-coor1
            difc = dif/wdn
        
            for p in range(1,wdn):
                widen[i-wdn+p][3:] = coor1+difc*p
    
    #fills between last with wdn
    for i, row in enumerate(widen):
        if i == wdn-1:
            coor1 = row [3:]
        elif i == yeniboy-1:
            coor2 = row [3:]
            dif = coor2-coor1
            difc = dif/wdn
            
            for p in range(wdn-1):
                widen[p][3:] = coor2-difc*(p+1)
        widen[i][0] = i +1  
    
    widen[:,1:3] = 1
    
    # tf_distance = int(yeniboy/tfd)
    
    for index,l in enumerate(widen):
        if index%tfd in [0,1,2]: #0 is the centers, rest is the side beads...
            l[1:3] =2
            widen[index]=l

    return widen
"""_________________________________________________________________________"""
def radius(n):
    """
    finding the necessary radius for given number of basepair to be consistent 
    with E.coli bp density. Where 1 bead represents 10bp. 
    Returns radius and necessary system volume respectively. 
    """
    import math
    
    pi = math.pi

    basepair = 4.6*10**6 #bp
    volume = 6.7*10**(-19) #m3
    bp_dens = basepair/volume #bp/m3
    realN = n*10 #my polymer bp
    sysVol = realN/bp_dens#bp/(bp/m3)
    sigma = 34*10**(-10) #1 sigma corresponds to 10 bp and 10bp length is 34 armstrong
    rreal = (3*sysVol/(16*pi))**(1/3) #2a^3 = m3, a = (m3/2)^(1/3)
    r = int(rreal/sigma+1)
    
    
    return r,sysVol
"""_________________________________________________________________________"""
def boundtf(fname,tfd,wdn=1):
    """
    Creates bounded transcription factors near promoter/binding sites
    Uses tfd(tf density) to create near promoter site
    tfd is related to promoter density.
    Return array atoms of bound transcription factors.
    """
    
    import numpy as np
    
    #uses steal function to obtain atoms
    atoms = steal(fname,tfd,wdn)
    
    nrow = len(atoms)
    n = nrow
    ncol = 6
    
    bound = np.zeros([int(nrow*3/tfd),ncol]) 
    
    i = 0
    for r in range(n-1):
        
        row = atoms[r]
        rowr = atoms[r+1]
        
        #bound atoms are created near promoters according to tfd
        if (r+1)%tfd == 0:
            bound[i]=[i+n+1,4,4,row[3]+0.6,row[4],row[5]]
            bound[i+1]=[i+n+2,5,5,row[3]+1.4,row[4],row[5]]
            bound[i+2]=[i+n+3,4,4,rowr[3]+0.6,rowr[4],rowr[5]]
            i +=3
    
    #bound atoms are stacked with DNA polymer atoms
    total = np.append(atoms,bound)
    nrow = int(len(total)/6)
    total = total.reshape(nrow,ncol)

    #array named total containing btf and DNA polymer atom coordinates is returned
    return total
"""_________________________________________________________________________"""
def freetf(um,index,n):
    """
    creates freee transcription factors within cellular boundries at random 
    positions with given density(um) in accordance with system volume.
    Return array of atoms of free transcription factors.
    """
    from random import random
    import numpy as np
    
    #atom and moleucule types for root and binding domains
    sap = 5
    typ = 3
    
    r,sysVol = radius(n)
    
    ttf = um
    avag = 6.022*(10**23) #avagdaro number mol
    m2l = 1000 #m^3 to liter
    m2u = 10**(-6) #meter to micrometer
    
    ftf = avag*m2l*m2u*sysVol*ttf #molarite to number
    
    ntf = int(ftf)
    
    kok2 = 2**(1/2)
    
    free = np.zeros([3*ntf,6])
    
    index = index+1
    
    #creating free TF at random positions within cell membrane
    for i in range (0,3*ntf,3):
        xcr = 4*r*random()-2*r
        ycr = 2*r*random()/kok2 -r/kok2
        zcr = 2*r*random()/kok2 -r/kok2
        
        free[i]=[index,typ,typ,xcr,ycr,zcr]
        free[i+1]=[index+1,sap,sap,xcr-0.33,ycr+0.28,zcr+0.31]
        free[i+2]=[index+2,typ,typ,xcr+0.25,ycr+0.33,zcr+0.38]
        index +=3
    #retutning a list of free TFs
    return free
"""_________________________________________________________________________"""
def cylinder(r,index,atom_type):
    """
    Creates a cylinder around given system using an r which could be obtained 
    using radius function or manually given.
    Return array of atoms of cylinder.
    """
    import math
    import numpy as np
    typ = atom_type
    cyl = np.array([])
    
    pir = r*3.14159
    pr = int(pir)
    
    #creating the atoms
    for xcr in range (-2*r,2*r+1):
        for x in range(-pr,pr+1):
    
            index +=1
            ycr = math.cos(x/r)*r
            zcr = math.sin(x/r)*r
            coor = [index,typ,typ,xcr,ycr, zcr]
            coor = np.array(coor)
            
            cyl = np.append(cyl,coor)
            
    
    ncol = 6
    row = len(cyl)/ncol
    nrow =  int(row)
    
    cyl = cyl.reshape(nrow,ncol)
    
    #returns an array with atoms of cylinder
    return cyl
"""_________________________________________________________________________"""
def cap(r,index,atom_type):
    """
    Creates caps around given system using an r which could be obtained 
    using radius function or manually given.
    Return array of atoms of caps.
    """
    import math
    import numpy as np
    

    pi = math.pi
    r = r
    gs = r
    angle_num=int(pi*gs/2)
    cap = np.array([])
    typ = atom_type
    ncol = 6
    
    #creating atoms of the one cap
    for xl in range (0,angle_num+1):
    
        angle = (pi/2)*xl/angle_num
        xcr = gs*math.cos(angle)
    
    
        r_new = math.sqrt(gs**2-xcr**2)
        num = int (2*pi*r_new)
        
    
        for i  in range (0,num,2):
            index +=1
    
            zcr = math.cos(2*pi/num*i)*r_new
            ycr = math.sin(2*pi/num*i)*r_new
            
            coor = [index,typ,typ,xcr+2*r+0.5,ycr, zcr]
            coor = np.array(coor)
            cap = np.append(cap,coor)
    
    
    r = -r
    gs = r
    angle_num = int(pi*gs/2)
    #creating atoms of the opposite cap
    for xl in range (0,angle_num-1,-1):
    
        angle = (pi/2)*xl/angle_num
        xcr = gs*math.cos(angle)
    
    
        r_new = math.sqrt(gs**2-xcr**2)
        num = int (2*pi*r_new)
        
    
        for i  in range (0,-num,-2):
            index +=1
    
            zcr = math.cos(2*pi/num*i)*r_new
            ycr = math.sin(2*pi/num*i)*r_new
            
            coor = [index,typ,typ,xcr+2*r-0.5,ycr, zcr]
            coor = np.array(coor)
            cap = np.append(cap,coor)
    
    row = len(cap)/6
    nrow = int(row)
    cap = cap.reshape(nrow,ncol)
    #returns an array with atoms of caps
    return cap
"""_________________________________________________________________________"""
def membrane(r,index,atom_type):
    """
    Utilizes cap and cyliner functions to create a full membrane to encapsulate
    the system.
    Return array of atoms of membrane.
    """
    import numpy as np
    
    cyl = cylinder(r,index,atom_type)
    index += len(cyl)
    
    cap1 = cap(r,index,atom_type)
    
    memb = np.append(cyl,cap1)
    memb = np.array(memb)
    ncol = 6
    nrow = int(len(memb)/ncol)
    memb = memb.reshape(nrow,ncol)
    #returns an array with atoms of membrane
    return memb
"""_________________________________________________________________________"""
def bonder(n,um,tfd):
    """
    creates bond among DNA monomers and TF monomers (1-2,2-3) to create "V"
    model.
    Returns an array of bonds.
    """
    import numpy as np

    #finds total number of bonds for bound&free tfs and DNA

    free = freetf(um,n,n)
    fr = int(len(free)/3)
    
    ntyp = 1
    
    #finds total number of transcription factors
    tfs = fr
    
    #total bond required
    num_bonds = n+tfs*2
    bonds = np.zeros([num_bonds,4])
    index = 0
    
    #bond type
    ntyp = 1
    #creating list for bonds of DNA polymer
    for i in range(n):
        index +=1
        if index < n:
            bonds[i] = [index,ntyp,index,index+1]
        elif index == n:
            bonds[i] = [index,ntyp,index,1]
            
    
    tftyp = 2
    indx = n
    b = 0
    #creating list for bonds of TFs
    for index in range(n+1,n+tfs*3+1,3):
        ilk = index
        iki = index+1
        uc = index+2
        bonds[indx] = [index+b,tftyp,ilk,iki]
        indx +=1
        bonds [indx] = [index+b+1,tftyp,iki,uc]
        indx +=1
        b-=1
    #returning the list of all the bonds
    return bonds
"""_________________________________________________________________________"""
def angler(n,um,tfd):
    """ 
    creates the angles between DNA and the transcription factors.
    Returns an array of angles.
    """
    import numpy as np
    
    
    
    free = freetf(um,n,n)
    fr = int(len(free)/3)
    
    ntyp = 1
    
    tfs = fr
    
    #total number of angles required
    num_angles = n+tfs
    angles = np.zeros([num_angles,5])
    index = 0
    
    #creating list for angles of DNA polymer
    for i in range(n):
        index += 1
        
        if i<n-2:
            angles[i]=[index,ntyp,index,index+1,index+2]
        elif i==n-2:
            angles[i]=[index,ntyp,index,index+1,1]
        elif i==n-1:
            angles[i]=[index,ntyp,index,1,2]
    
    tftyp = 2
    indx = index +1
    #creating list for angles of TFs
    for j in range(tfs):
        index = index +1
        angles[n+j]=[index,tftyp,indx+j,indx+j+1,indx+j+2]
        indx = indx +2
    #returning the list of all the angles 
    return angles
"""_________________________________________________________________________"""
def buildNwrite(um,filetoread,filetowrite,tfd=40,wdn=1):
    """
    takes all necessary arguments to use other functions in this module
    builds all arrays and lists and write them into a data file with given name
    Does not return anything
    """
    import numpy as np
    
    tf = tfd
    pos =  steal(filetoread,tfd,wdn) #steals only DNA
    ps = len(pos) #length of the DNA polymer array
    n = ps #again the length equal n
    r,sysVol = radius(n)
    
    ftf = freetf(um,ps,n)
    ft = len(ftf) 
    index = ft + ps
    
    mem = membrane(r,index,6)
    angles = angler(n,um,tf)
    bonds = bonder(n,um,tf)
    
    atoms1 = np.append(pos,ftf)
    atoms = np.append(atoms1,mem)
    
    boy = len(atoms)
    ncol = 6
    nrow = int(boy/ncol)
    
    atoms = atoms.reshape(nrow,ncol)
    
    #open the file to write
    ll = open(filetowrite,"w")
    
    #starts writing here
    ll.write("\n\n")
    
    #atoms,bonds,angles declared
    num_atoms = str(nrow)
    ll.write(num_atoms+" atoms\n")
    ll.write("6 atom types\n")
    bnds = str(len(bonds))
    ll.write(bnds+" bonds\n")
    ll.write("2 bond types\n")
    angls = str(len(angles))
    ll.write(angls+" angles\n")
    ll.write("2 angle types\n\n")
    
    #boundaries declared
    x = 2.2324242
    ll.write(str(-3*r-x)+" "+str(3*r+x)+" xlo xhi\n")
    ll.write(str(-r-x)+" "+str(r+x)+" ylo yhi\n")
    ll.write(str(-r-x)+" "+str(r+x)+" zlo zhi\n\n")
    
    #masses declared
    ll.write("Masses\n\n")
    ll.write("1 1\n")   
    ll.write("2 1\n")
    ll.write("3 2\n")
    ll.write("4 2\n")
    ll.write("5 2\n")
    ll.write("6 1\n\n")
    
    #pair coeffs declared (to be overwritten in input file)
    ll.write("Pair Coeffs # lj/cut\n\n")
    
    ll.write("1 12 1\n")   
    ll.write("2 12 1\n")
    ll.write("3 12 1\n")
    ll.write("4 12 1\n")
    ll.write("5 12 1\n")
    ll.write("6 12 1\n\n")
    
    #bond coeffs declared
    ll.write("Bond Coeffs # fene\n\n")
    
    ll.write("1 30 1.5 1 1\n")   
    ll.write("2 30 1.0 0.71 0.71\n\n")
    
    #angle coeff declared
    ll.write("Angle Coeffs # harmonic\n\n")
    
    ll.write("1 1 180.0\n")
    ll.write("2 20 90\n\n")
    
    #writing atoms to the file
    ll.write("Atoms # angle\n\n")
    
    for row in atoms:
        for i in range (6):
            if i<3:
                ii = int(row[i])
                ll.write(str(ii)+" ")
            else:
                ll.write(str(row[i])+" ")
        ll.write("\n")
    
    #writing bonds to the file
    ll.write("\nBonds\n\n")
    
    for row in bonds:
        for i in range (4):
                ii = int(row[i])
                ll.write(str(ii)+" ")
        ll.write("\n")
    
    #writing bonds to the file
    ll.write("\nAngles\n\n")
    for row in angles:
        for i in range (5):
                ii = int(row[i])
                ll.write(str(ii)+" ")
        ll.write("\n")
    #this function returns nothing
    return

if __name__ == "__main__":

    from timeit import default_timer as timer
    
    start = timer()
    
    buildNwrite(60,"data.col2","data.denge", tfd = 40, wdn = 4)
    finish = timer()
    diff = finish-start
    print(f'Time Elapsed: {diff:.2f} secnonds')
    