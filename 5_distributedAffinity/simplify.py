import polars as pl

data = pl.read_csv("data/duration_cont.csv")
data = data.drop(pl.col("3.50_60"))
# get the sum of rows
data = data.with_columns(pl.sum_horizontal(pl.all()).alias("total"))
print(data.height)
# get only the rows with total > 0
data = data.filter(pl.col("total") > 0)

print(data.height)
print(data.tail())

# replace 0s with null

data = data.with_columns(
    [
        pl.when(pl.col(col) == 0).then(pl.lit(None)).otherwise(pl.col(col)).alias(col)
        for col in data.columns
    ]
)

data = data.drop(pl.col("total"))
data = data.with_columns(
    pl.int_range(0, data.height, 1).add(1).alias("duration")
    ).select(pl.col("duration"), pl.all().exclude("duration"))
print(data.tail())


data.write_csv("data/durations_cleaned.csv")
