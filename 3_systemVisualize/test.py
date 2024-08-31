import polars as pl

df = pl.DataFrame()
df.with_columns(
    (pl.col("mean_size") - pl.col("std_size")).alias("cluster_size_minus_std"),
    (pl.col("mean_size") + pl.col("std_size")).alias("cluster_size_plus_std"),
    (
        pl.col("mean_size") - 1.96 * (pl.col("std_size") / pl.col("sample_size").sqrt())
    ).alias("confidence_95_low"),  # z = 1.96 for CI 95
    (
        pl.col("mean_size") + 1.96 * (pl.col("std_size") / pl.col("sample_size").sqrt())
    ).alias("confidence_95_high"),
)
