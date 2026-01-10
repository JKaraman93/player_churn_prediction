from pyspark.sql import functions as F


def generate_transactions(players_df):
    df = (
        players_df
        .sample(fraction=0.4)
        .withColumn("transaction_id", F.expr("uuid()"))
        .withColumn(
            "transaction_type",
            F.expr("CASE WHEN rand() < 0.7 THEN 'deposit' ELSE 'withdrawal' END")
        )
        .withColumn("amount", F.round(F.rand() * 200, 2))
        .withColumn(
            "success_flag",
            F.expr("CASE WHEN rand() < 0.95 THEN true ELSE false END")
        )
        .withColumn("transaction_ts", F.current_timestamp())
        .select(
            "transaction_id",
            "player_id",
            "transaction_ts",
            "transaction_type",
            "amount",
            "success_flag"
        )
    )
    return df