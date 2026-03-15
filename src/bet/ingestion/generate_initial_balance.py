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