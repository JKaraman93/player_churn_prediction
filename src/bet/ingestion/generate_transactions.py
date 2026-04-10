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

from pyspark.sql import DataFrame, functions as F
from bet.utils.config import DataGenConfig
from bet.utils.logging_utils import get_logger

logger = get_logger(__name__)


def generate_financial_transactions(players_df: DataFrame, config: DataGenConfig = None) -> DataFrame:
    """
    Generate synthetic financial transactions with deposits and withdrawals.
    
    Creates transactions with:
    - 60% of players sampled
    - 70% deposits, 30% withdrawals
    - Random amounts 0-200 currency units
    - 90% success rate for all transactions
    - Timestamps uniformly distributed through date range
    
    Args:
        players_df: DataFrame with player profiles
        config: Optional DataGenConfig instance (creates default if not provided)
        
    Returns:
        DataFrame with financial transaction records
    """
    if config is None:
        config = DataGenConfig()
    
    logger.info(f"Generating financial transactions from {int(players_df.count() * 0.6)} players")
    start_ts = F.to_timestamp(
        F.concat(F.lit(config.start_date), F.lit(" 00:00:00")),
        "yyyy-MM-dd HH:mm:ss"
    )

    end_ts = F.to_timestamp(
        F.concat(F.lit(config.end_date), F.lit(" 23:59:59")),
        "yyyy-MM-dd HH:mm:ss"
    )
    
    df = (players_df
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
            (F.rand() * (end_ts.cast("long") - start_ts.cast("long")))
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
    
    logger.info(f"Generated {df.count()} financial transactions")
    return df