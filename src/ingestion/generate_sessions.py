from pyspark.sql import functions as F

def generate_gameplay_sessions(players_df, spark, config): 
    date_range = spark.sql(
        f"SELECT explode(sequence(to_date('{config.start_date}'), to_date('{config.end_date}'), interval 1 day)) AS event_date"
    )

    df = players_df.crossJoin(date_range)
    df = (
        df
        .withColumn(
        "daily_sessions",
        F.when(
            F.col("lifecycle_stage") == "engaged",
            F.floor(-F.log(1 - F.rand(seed=42)) * F.lit(config.active_lambda))
        ).when(
            F.col("lifecycle_stage") == "at_risk",
            F.floor(-F.log(1 - F.rand(seed=42)) * F.lit(config.at_risk_lambda))
        ).otherwise(F.lit(0))
    )
        .filter(F.col("daily_sessions") > 0)
        .withColumn("session_seq", F.explode(F.sequence(F.lit(1), F.col("daily_sessions"))))
        .withColumn("session_id", F.expr("uuid()"))
        .withColumn("session_ts",
            (F.col("event_date").cast("timestamp").cast("long") +
                (F.rand() * 86400).cast("int")
            ).cast("timestamp"),)
        .withColumn("game_id", F.concat(F.lit("G"), (F.rand() * 200).cast("int")))
        .withColumn("session_duration_sec", (F.rand() * 3600).cast("int"))
        .withColumn("bet_count", (F.rand() * 20).cast("int"))
        .withColumn("total_bet_amount", F.round(F.rand() * 100, 2))
        .withColumn("total_win_amount", F.round(F.rand() * 120, 2))
        .select(
            "session_id",
            "player_id",
            "game_id",
            F.col("session_ts").alias("session_date"),
            "session_duration_sec",
            "bet_count",
            "total_bet_amount",
            "total_win_amount",
            #"device_type"
        )
    )
                
    return df
    