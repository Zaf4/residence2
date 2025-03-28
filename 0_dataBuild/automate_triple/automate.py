from zkdatabuilder import zkdatabuilder
import timeit
import os
import shutil

def autofold(workID, um_list, kT_list = [12], interaction_type = "NSI", replica_N = 1, run=True, read=True):

    system = "4X"
        
    replic = replica_N

    init = os.getcwd()

    location_list = []
    start = timeit.default_timer()
    dep_folder = init +"\\dependencies"
    for i in system:
        os.chdir(init)
        if not os.path.exists(i):
            os.mkdir(i+workID)
        
        for k in kT_list:
            os.chdir(init+"\\" + i+workID)
            if not os.path.exists(k):
                os.mkdir(k)

            for u in um_list:
                os.chdir(init+"\\"+i+workID+"\\"+k)
                if not os.path.exists(u):
                    os.mkdir(u)
                
                if u !="0":
                    replica_N = replic
                else:
                    replica_N = 1


                os.chdir(init+"\\"+i+workID+"\\"+k+"\\"+u)
                subf = "r"+str(r+1)
                if not os.path.exists(subf):
                    os.mkdir(subf)
                final_dest = init+"\\"+i+workID+"\\"+k+"\\"+u+"\\"+subf
                location_list.append(i+workID+"\\"+k+"\\"+u+"\\"+subf)
                os.chdir(final_dest)
                zkdatabuilder.buildNwrite(um = int(u),
                        filetoread=init+"\\template\\data.col2", 
                        filetowrite=final_dest+"\\data.denge",
                        tfd=100,
                        wdn=5)
                #copy the files
                if u == "0":
                    main_input = dep_folder+"\\"+i+"\\0\\in.main"
                else:
                    main_input = dep_folder+"\\"+i+"\\in.main"
                shutil.copy(main_input,final_dest,follow_symlinks=True)
                
                for f in files_list:
                    ftocopy = dep_folder+"\\"+f
                    shutil.copy(ftocopy,final_dest,follow_symlinks=True)

                timediff = timeit.default_timer()-start
                print(os.getcwd(),timediff)       
                    
    os.chdir(init)

    if run == True:
        ll = open("script_to_run.txt","w")
        for l in location_list:
            l = l.replace('\\','/')
            ll.write("cd "+l+"\n")
            ll.write("sbatch submit.sh\n")
            ll.write("cd ..\ncd ..\ncd ..\ncd ..\n\n")
        ll.close()

    if read == True:
        gg = open("script_to_read.txt","w")
        for l in location_list:
            l = l.replace('\\','/')
            gg.write("cd "+l+"\n")
            gg.write("python3 dr.py\n")
            gg.write("cd ..\ncd ..\ncd ..\ncd ..\n\n")
        gg.close()
    return


if __name__ == '__main__':
    files_list = [  "in.denge",
                    "submit.sh",
                    "dr.py",
                    ]
    interactions = "NSI"
    uMs = ["68","108"]
    kTs = ["4-4-6"]
    workid = "80-120um"

    autofold(workid,uMs,kTs,interaction_type=interactions,replica_N=2)


