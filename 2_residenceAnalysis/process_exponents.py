import numpy as np
import pandas as pd
from lets_plot import *
LetsPlot.setup_html()
import os

# Load data and melt the dataframe
df = pd.read_csv("data/exponents.csv", encoding='ISO-8859-1')
dfm = pd.melt(df, id_vars=['equation','energy','concentration'], var_name='exponent', value_name='value')
dfm['value'] = dfm['value']+2
# Remove rows with NaN values
dfm.dropna(inplace=True)
dfm.to_csv("data/exponents_melted.csv", encoding='ISO-8859-1')

# Split data into coeff and tau dataframes
df_tau = dfm[dfm['exponent'].str.contains('tau')]
df_coeff = dfm[dfm['exponent'].str.contains('coeff')]

# Plot the data
p1 = (
    ggplot(df_tau, aes(x='concentration', y='value', fill='energy'))+
    geom_bar(color='black',stat='identity', position='dodge')+
    scale_fill_hue()+
    ylab('Tau')+
    xlab('Concentration')+
    scale_y_log2()+
    theme(exponent_format='pow')
    )


# Plot the data
p2 = (
    ggplot(df_coeff, aes(x='concentration', y='value', fill='energy'))+
    geom_bar(color='black',stat='identity', position='dodge')+
    scale_fill_viridis()+
    ylab('Coefficient')+
    xlab('Concentration')+
    scale_y_log2()
    
    )
p = gggrid([p1,p2])+ggsize(1600,800)
p.to_html("exponents.html")



if __name__ == "__main__":
    # Load data
    #changing working directory to current directory name
    os.chdir(os.path.dirname(__file__))