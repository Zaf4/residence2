import os

import pandas as pd
from lets_plot import (
    LetsPlot,
    aes,
    element_blank,
    geom_bar,
    gggrid,
    ggplot,
    ggsize,
    scale_fill_hue,
    scale_fill_viridis,
    scale_y_log2,
    scale_y_log10,
    theme,
    theme_classic,
    xlab,
    ylab,
    element_text
)

LetsPlot.setup_html()


# Load data and melt the dataframe
if os.path.exists("data/exponents_melted.csv"):
    dfm = pd.read_csv("data/exponents_melted.csv", encoding="ISO-8859-1")
else:
    df = pd.read_csv("data/exponents.csv", encoding="ISO-8859-1")
    dfm = pd.melt(
        df,
        id_vars=["equation", "energy", "concentration"],
        var_name="exponent",
        value_name="value",
    )
    dfm["value"] = dfm["value"] + 2
    # Remove rows with NaN values
    dfm.dropna(inplace=True)
    dfm.to_csv("data/exponents_melted.csv", encoding="ISO-8859-1", index=False)

# Split data into coeff and tau dataframes
df_tau = dfm[dfm["exponent"].str.contains("tau")]
df_coeff = dfm[dfm["exponent"].str.contains("coeff")]

# Plot the data
p1 = (
    ggplot(df_tau, aes(x="concentration", y="value", fill="energy"))
    + geom_bar(color="black", stat="identity", position="dodge")
    + scale_fill_hue()
    + ylab("Tau")
    + xlab("Concentration")
    + scale_y_log2()
    + theme_classic()
    + theme(exponent_format="pow", legend_title = element_blank())
    
)


# Plot the data
p2 = (
    ggplot(df_coeff, aes(x="concentration", y="value", fill="energy"))
    + geom_bar(color="black", stat="identity", position="dodge")
    + scale_fill_viridis()
    + ylab("Coefficient")
    + xlab("Concentration")
    + scale_y_log2()
    + theme_classic()
    + theme(exponent_format="pow",legend_title = element_blank())

)

p10 = gggrid([p1, p2]) + ggsize(1600, 800)
p10.to_html("figures/exponents_by_um.html")


# Plot the data
p3 = (
    ggplot(df_tau, aes(x="equation", y="value", fill="energy"))
    + geom_bar(color="black", stat="identity", position="dodge")
    + scale_fill_hue()
    + ylab("Tau")
    + xlab("Equation")
    + scale_y_log2()
    + theme_classic()   
    + theme(exponent_format="pow", legend_title = element_blank())
    
)


# Plot the data
p4 = (
    ggplot(df_coeff, aes(x="equation", y="value", fill="energy"))
    + geom_bar(color="black", stat="identity", position="dodge")
    + scale_fill_viridis()
    + ylab("Coefficient")
    + xlab("Equation")
    + scale_y_log10(format=".0~e")
    + theme_classic()  
    + theme(exponent_format="pow",legend_title = element_blank())
     
)

p20 = gggrid([p3, p4]) + ggsize(1600, 600)
p20.to_html("figures/exponents_by_equation.html")
p20.to_svg("../Figures/SI-fig2.svg")

# Plot the data
p5 = (
    ggplot(df_tau, aes(x="energy", y="value", fill="concentration"))
    + geom_bar(color="black", stat="identity", position="dodge")
    + scale_fill_hue()
    + ylab("Tau")
    + xlab("Equation")
    + scale_y_log2()
    + theme_classic()  
    + theme(legend_title = element_blank())
     
)


# Plot the data
p6 = (
    ggplot(df_coeff, aes(x="energy", y="value", fill="concentration"))
    + geom_bar(color="black", stat="identity", position="dodge")
    + scale_fill_viridis()
    + ylab("Coefficient")
    + xlab("Equation")
    + scale_y_log10()
    + theme(exponent_format="pow", legend_title = element_blank())
    + theme_classic()   
)


p30 = gggrid([p5, p6]) + ggsize(1200, 600)
p30.to_html("figures/exponents_by_kT.html")

if __name__ == "__main__":
    # Load data
    # changing working directory to current directory name
    os.chdir(os.path.dirname(__file__))

