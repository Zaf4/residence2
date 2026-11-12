These are the distributions data from the the simulations run under the conditions:


Nonspecific binding energies of `KTS = ['1.00','2.80','3.00','3.50','4.00']` kTs.
Concentrations of `UMS = ['10','20','40','60']` micromolars.

Where __specific binding energy is fixed and 4 KT__ for every case in distributions file here.

There are much less specific binding sites (where proteins bind specifically) compared to nonspecific sites (where proteins bind nonspecifically).

The "count" column tell us the number of times that the proteins stayed on DNA for the given "duration".

IN short,
--------

Any __SP__ suffix/prefix means that only the interactions __between specific binding sites__ and the proteins are taken into account.
Whereas any __NS__ suffix/prefix means that only the interactions __between nonspecific sites__ and the proteins are taken into account.

**FULL_** prefixed files only merges **SP_** prefixed and **NS_** prefixed files.

Formats
-------
The same files are both in **csv** and **feather** format.

In case reading csv leads to the loss of metadata feather could be used instead.

Tidy vs Raw
-----------
There is both 
    - raw/matrix style dataframes with *.raw.* naming style.
    - tidy dataframes with *.tidy.* naming style.


### Tidy
with pandas
```
df = pd.read_csv("data/csv/FULL_distributions.tidy.csv")
```

with polars
```
df = pl.read_csv("data/csv/FULL_distributions.tidy.csv")
```

### Raw

or if you prefer the matrix format
with pandas
```
df = pd.read_csv("data/csv/FULL_distributions.raw.csv")
```
with polars
```
df = pl.read_csv("data/csv/FULL_distributions.tidy.csv")
```


the dataframe will have all KTS, UMS, and cases (SP/NS).  
Or you can directly use: **SP_**  or **NS_** versions.