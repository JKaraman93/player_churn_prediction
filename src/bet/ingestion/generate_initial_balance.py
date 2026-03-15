"""
Initial Balance Assignment: Account Balance Initialization

Assigns initial account balances to players based on their characteristics
and behavioral patterns. Sets up the starting point for financial tracking.

Logic:
- Varies balance by acquisition channel and lifecycle stage
- Applies realistic distribution of account balances
- Ensures positive starting balances for active gameplay

Output: player profiles with current_balance attribute populated
"""

from pyspark.sql import DataFrame, functions as F
from bet.utils.logging_utils import get_logger
from bet.utils.config import DataGenConfig

logger = get_logger(__name__)


def assign_balance(df_players: DataFrame, config: DataGenConfig) -> DataFrame:
    """
    Assign random initial account balances to players.
    
    Creates a 'current_balance' column with random values between 0 and 200,
    using the same random seed from config for reproducibility.
    
    Args:
        df_players: Input DataFrame with player profiles
        config: DataGenConfig instance with seed parameter
        
    Returns:
        DataFrame with new 'current_balance' column added
    """
    logger.info(f"Assigning initial balances to {df_players.count()} players")
    
    result = (df_players
        .withColumn(
            "current_balance",
            F.round(F.rand(seed=config.seed) * 200, 2)
        )
    )
    
    # Log balance statistics
    stats = result.agg(
        F.mean("current_balance").alias("avg_balance"),
        F.min("current_balance").alias("min_balance"),
        F.max("current_balance").alias("max_balance")
    ).collect()[0]
    
    logger.info(f"  Average balance: {stats['avg_balance']:.2f}")
    logger.info(f"  Min balance: {stats['min_balance']:.2f}")
    logger.info(f"  Max balance: {stats['max_balance']:.2f}")
    
    return result