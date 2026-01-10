from pyspark.sql import functions as F

def assign_balance(df_players, config):
    return (
        df_players
        .withColumn(
            "balance",
            F.round(F.rand(seed=config.seed) * 200 ,2
        )
    )
    )