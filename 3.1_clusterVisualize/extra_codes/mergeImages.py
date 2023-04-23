from PIL import Image

def create_row(casename:str=''):
    ov = Image.open(f'./figures/overview{casename}.png')
    pr = Image.open(f'./figures/proteins{casename}.png')
    cl = Image.open(f'./figures/clusters{casename}.png')
    
    w,h = cl.size  
    
    
    row = Image.new(mode='RGBA',size=(w*3,h))
    
    for i,image in enumerate((ov,pr,cl)):
        row.paste(image,(i*w,0))
        
   
    return row

def merge_rows(cases:list):
    
    samplerow = create_row(cases[0])
    w,h = samplerow.size
    whole = Image.new(mode='RGBA',size=(w,h*len(cases)))
    
    for i,case in enumerate(cases):
        row = create_row(case)
        whole.paste(row,(0,i*h))
        
        
    return whole


if __name__ == '__main__':
    cases = [100,280,300,350,400]
    w = merge_rows(cases)
    w = w.resize((int(w.size[0]/4),int(w.size[1]/4)))
    w.show()
    w.save('./figures/full.png',optimize=True)