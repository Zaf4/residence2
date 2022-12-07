import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


colors = ['#2456E5','#C70039','#581845']

def hex2rgb(hex):
  return tuple(int(hex[i:i+2], 16) for i in (1, 3, 5))

def hexlist2rgb(hex_colors):
    return [hex2rgb(hex_color) for hex_color in hex_colors ]



def generate_palette(colors:list,num_colors:int):
    rgb_colors = np.array(hexlist2rgb(colors))
    r,g,b = rgb_colors[:,0],rgb_colors[:,1],rgb_colors[:,2]
    #initing palette array --RGBA
    palette = np.zeros([num_colors,3])
    if len(colors)==2:
        #reds
        palette[:,0] = np.linspace(r[0],r[-1],num_colors)

        #greens
        palette[:,1] = np.linspace(g[0],g[-1],num_colors)

        #blues
        palette[:,2] = np.linspace(b[0],b[-1],num_colors)

    
    elif len(colors)==3:
        mid = int(num_colors/2)#middle of the array
        #reds
        palette[:mid,0] = np.linspace(r[0],r[1],mid)
        palette[mid:,0] = np.linspace(r[1],r[2],num_colors-mid)
        #greens
        palette[:mid,1] = np.linspace(g[0],g[1],mid)
        palette[mid:,1] = np.linspace(g[1],g[2],num_colors-mid)
        #blues
        palette[:mid,2] = np.linspace(b[0],b[1],mid)
        palette[mid:,2] = np.linspace(b[1],b[2],num_colors-mid)
        
    return palette/256



deep = generate_palette(['#2456E5','#C70039','#581845'], 20)

deeper = sns.color_palette(deep)    
    
ax = plt.figure()
font = {'family': 'Arial',
        'weight': 'light',
        'size': 14,
        }

sns.set(style='ticks',
    rc = {'figure.figsize':(11,9),
              'font.weight':'light',
              'font.family':'sans-serif',
              'axes.spines.top':'False',
              'axes.spines.right':'False',
              'ytick.minor.size':'0',
              'xtick.minor.size':'0',
              'ytick.major.size':'10',
              'xtick.major.size':'10'
              
              }
        )

data = pd.read_csv('./data/durations_minimized.csv',index_col=None)
sns.scatterplot(data,palette=deeper,markers=(['o']*data.shape[1]),
                s=50,edgecolor='black',linewidth=0.25)

plt.yscale('log')
plt.xscale('log')
plt.xlim([0.7,3e3])
plt.ylim([0.5,1e6])
plt.xlabel('Duration (a.u.)',fontdict=font)
plt.ylabel('Occurence',fontdict=font)
plt.legend(['nope'])
    
plt.show()

