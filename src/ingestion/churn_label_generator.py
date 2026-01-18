# src/ingestion/generate_churn_labels.py

from pyspark.sql import functions as F

def generate_churn_labels(sessions_df, config):
    last_activity = (
        sessions_df
        .groupBy("player_idx")
        .agg(F.max("session_date").alias("last_session_date"))
    )

    labels = (
        last_activity
        .withColumn(
            "churn_7d",
            F.expr(
                f"datediff('{config.end_date}', last_session_date) >= {config.churn_inactivity_days}"
            )
        )
        .withColumn("reference_date", F.lit(config.end_date))
    )

    return labels
