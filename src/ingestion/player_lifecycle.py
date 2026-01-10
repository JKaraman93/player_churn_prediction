from pyspark.sql import functions as F

def assign_lifecycle(df_players):
    return (
        df_players
        .withColumn(
            "lifecycle_stage",
            F.expr("""
                CASE
                    WHEN rand() < 0.6 THEN 'engaged'
                    WHEN rand() < 0.85 THEN 'new'
                    ELSE 'at_risk'
                END
            """)
        )
    )
