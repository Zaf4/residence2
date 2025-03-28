import db_triple
import timeit
import os
import shutil

def setKT(ftoread, ftowrite, NSI, SPI):
    f = open(ftoread)
    lines = f.readlines()
    f.close()
    
    ns=lines.index('pair_coeff 1 3 NSI 0.714 2.5\n')
    lines[ns] = f'pair_coeff 1 3 {NSI} 0.714 2.5\n'
    sp=lines.index('pair_coeff 2 3 SPI 0.714 2.5\n')
    lines[sp] = f'pair_coeff 2 3 {SPI} 0.714 2.5\n'
    
    t = open(ftowrite,'w')
    for l in lines:
        t.write(l)
        
    t.close()
    return


def autofold(workID, um_list, kT_list, file_list, run=True, read=True):

    system = ["4X"]
    SPI = "4.00"

    init = os.getcwd()

    location_list = []
    start = timeit.default_timer()
    dep_folder = init +"\\template"

    #system
    for i in system:
        os.chdir(init)
        if not os.path.exists(i+workID):
            os.mkdir(i+workID)
        
        #affinities
        for k in kT_list:
            os.chdir(init+"\\" + i+workID)
            if not os.path.exists(k):
                os.mkdir(k)

            setKT(dep_folder+"\\in_template.main",dep_folder+"\\in.main",k,SPI) #in.main reconfigured

            #concentration
            for u in um_list:
                os.chdir(init+"\\"+i+workID+"\\"+k)
                if not os.path.exists(u):
                    os.mkdir(u)

                os.chdir(init+"\\"+i+workID+"\\"+k+"\\"+u)
                final_dest = init+"\\"+i+workID+"\\"+k+"\\"+u
                location_list.append(i+workID+"\\"+k+"\\"+u)
                os.chdir(final_dest)

                db_triple.buildNwrite(um = int(u),
                        filetoread=init+"\\template\\data.col2", 
                        filetowrite=final_dest+"\\data.denge",
                        tfd=40,
                        wdn=4)

                #files copied
                for f in files_list:
                    ftocopy = dep_folder+"\\"+f
                    shutil.copy(ftocopy,final_dest,follow_symlinks=True)

                timediff = timeit.default_timer()-start
                print(os.getcwd(),timediff)       
                    
    os.chdir(init)

    if run == True:
        ll = open("run.txt","w")
        for l in location_list:
            l = l.replace('\\','/')
            ll.write("cd "+l+"\n")
            ll.write("sbatch submit.sh\n")
            ll.write("cd \n")
        ll.close()

    if read == True:
        gg = open("restate.txt","w")
        for l in location_list:
            l = l.replace('\\','/')
            gg.write("cd "+l+"\n")
            gg.write("python3 restate.py\n")
            gg.write("cd \n")
        gg.close()

        pp = open("dist.txt","w")
        for l in location_list:
            l = l.replace('\\','/')
            pp.write("cd "+l+"\n")
            pp.write("python3 dist.py\n")
            pp.write("cd \n")
        pp.close()
    return


if __name__ == '__main__':
    files_list = [  "in.main",
                    "in.denge",
                    "submit.sh",
                    "restate.py",
                    "dist.py"]

    uMs = ["80","120"]
    kTs = ["2.80","3.00","3.50"]
    workid = "_rest"

    autofold(workid,uMs,kTs,files_list)


