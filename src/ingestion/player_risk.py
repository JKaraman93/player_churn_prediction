from pyspark.sql import functions as F

def assign_risk(df_players):
    return (
        df_players
        .withColumn(
            "risk_segment", 
            F.when( F.col('lifecycle_stage')=='new', F.lit("unknown"))
            .when(F.rand()<0.6, F.lit("low"))
            .when(F.rand()<0.9, F.lit("medium"))
            .otherwise(F.lit('high'))
        )
    )