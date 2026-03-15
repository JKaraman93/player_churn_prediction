"""
Financial Transaction Generation: Synthetic Payment Data

Generates synthetic financial transactions (deposits and withdrawals)
with realistic amounts, success rates, and timestamps.

Generated attributes:
- player_id: Reference to player profile
- transaction_id: Unique transaction identifier
- transaction_type: 'deposit' or 'withdrawal'
- amount: Transaction amount in currency units
- success_flag: Boolean indicating if transaction succeeded
- transaction_ts: Timestamp of transaction

Used for money flow and financial health features in Gold layer.
"""

from pyspark.sql import functions as F
import bet.utils.config as config




def generate_financial_transactions(players_df):
    config_ = config.DataGenConfig()
    start_ts = F.to_timestamp(
        F.concat(F.lit(config_.start_date), F.lit(" 00:00:00")),
        "yyyy-MM-dd HH:mm:ss"
    )

    end_ts = F.to_timestamp(
        F.concat(F.lit(config_.end_date), F.lit(" 23:59:59")),
        "yyyy-MM-dd HH:mm:ss"
    )
    
    df = (
        players_df
        .sample(fraction=0.6)
        .withColumn("transaction_id", F.expr("uuid()"))
        .withColumn(
            "transaction_type",
            F.expr("CASE WHEN rand() < 0.7 THEN 'deposit' ELSE 'withdrawal' END")
        )
        .withColumn("amount", F.round(F.rand() * 200, 2))
        .withColumn(
            "success_flag",
            F.expr("CASE WHEN rand() < 0.90 THEN true ELSE false END")
        )
        .withColumn("transaction_ts", (
        start_ts.cast("long") +
        (
            F.rand() *
            (end_ts.cast("long") - start_ts.cast("long"))
        )
    ).cast("timestamp"))
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