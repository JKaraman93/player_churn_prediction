"""
Bronze Layer Generation: Synthetic Data Creation

This module generates synthetic raw data for the entire pipeline.
It creates realistic player profiles, gaming sessions, and financial transactions
based on configured parameters.

Generated data:
1. players: Player profiles with lifecycle stages and risk segments
   - Demographics (country, age, acquisition channel)
   - Account information (registration date, balance)
   - Lifecycle classification (new, engaged, at_risk, churned)
   - Risk segmentation based on betting patterns

2. sessions: Player gaming sessions
   - Session metadata (date, duration)
   - Betting activity and game outcomes
   - Deposits and withdrawals during sessions

3. transactions: Financial transactions
   - Transaction types (deposit, withdrawal)
   - Success/failure flags
   - Transaction amounts and timestamps

Configuration:
- Uses DataGenConfig from bet.utils.config
- Configurable date range, number of players, and behavioral parameters

Outputs:
- Parquet tables in data/bronze/ directory
- Starting point for medallion architecture data pipeline
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from bet.utils.config import DataGenConfig
from bet.utils.logging_utils import get_logger
from bet.ingestion.generate_players import generate_player_profiles
from bet.ingestion.player_lifecycle import assign_lifecycle
from bet.ingestion.generate_sessions import generate_gameplay_sessions
from bet.ingestion.generate_transactions import generate_financial_transactions
from bet.ingestion.generate_initial_balance import assign_balance
from bet.ingestion.player_risk import assign_risk

logger = get_logger(__name__)


def main() -> None:
    """
    Generate Bronze layer with synthetic raw data.
    
    Orchestrates the complete data generation pipeline:
    1. Creates player profiles with demographics
    2. Assigns lifecycle stages (new, engaged, at_risk, churned)
    3. Assigns risk segments (unknown, low, medium, high)
    4. Initializes account balances
    5. Generates gaming sessions with timestamps
    6. Generates financial transactions (deposits/withdrawals)
    7. Writes all tables to data/bronze/ directory
    
    Returns:
        None
    """
    logger.info("Starting Bronze layer generation")
    
    config = DataGenConfig()
    
    # Create Spark session with local mode
    spark = SparkSession.builder.master("local[*]").appName('bronze_generation').getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    logger.info(f"Generating data for {config.num_players} players from {config.start_date} to {config.end_date}")
    
    # Generate core player data
    df_players = generate_player_profiles(spark, config)
    df_players = assign_lifecycle(df_players)
    df_players = assign_risk(df_players)
    df_players = assign_balance(df_players, config)
    
    # Generate behavioral data
    df_sessions = generate_gameplay_sessions(df_players, spark, config)
    df_money_transactions = generate_financial_transactions(df_players, config)
    
    # Write to bronze layer
    logger.info("Writing tables to data/bronze/")
    df_players.write.mode("overwrite").parquet("./data/bronze/players")
    df_money_transactions.write.mode("overwrite").parquet("./data/bronze/transactions")
    df_sessions.write.mode("overwrite").partitionBy("session_date").parquet("./data/bronze/sessions")
    
    logger.info("Bronze layer generation completed successfully")


if __name__ == "__main__":
    main()

