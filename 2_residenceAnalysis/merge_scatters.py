from PIL import Image
import os 

def merge_graphs(rows:list,cols:list):
    nrow,ncol = len(rows),len(cols)

    w,h = Image.open(f'./scatters/{rows[1]}_{cols[1]}.png').size

    multi = Image.new(mode='RGBA',size=(w*ncol,h*nrow))
    
    for i,keyword in enumerate(cols):
        for j,eqname in enumerate(rows):
        
            graph = Image.open(f'./scatters/{eqname}_{keyword}.png')

            multi.paste(graph,(i*w,j*h))


    return multi


if __name__ == '__main__':
    eqnames = ['ED','DED','TED','QED','PED','Powerlaw']
    ums = ['10','20','40','60']
    kts = ['100','280','300','350','400']


    multi_um = merge_graphs(eqnames,ums)
    multi_kt = merge_graphs(eqnames,kts)

    if not os.path.exists('./mergedImages'):
        os.mkdir('mergedImages')

    multi_um.save('./mergedImages/MultiUM.png')
    multi_kt.save('./mergedImages/MultiKT.png')
    #multi_kt.show()
    #multi_um.show()