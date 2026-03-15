"""
Inference Data Preparation: Feature Engineering for Predictions

This module prepares feature vectors for churn risk inference on a given date.
It loads Silver-layer data (sessions, transactions, money events) and computes
rolling 7-day and 30-day behavioral aggregates with strict time causality.

Key transformations:
- Aggregates betting patterns, win/loss statistics over rolling windows
- Computes session counts, durations, and user engagement metrics
- Applies player lifecycle segmentation
- Ensures all features use data only up to (and including) the inference date

Designed for production inference to guarantee feature consistency with training.

Inputs:
- test_date: The date for which to generate features (YYYY-MM-DD format)
- Silver layer parquet tables (players, sessions, transactions, money_events)

Outputs:
- Feature vector ready for model inference
- Player snapshot with static attributes
- Behavioral features with 7-day and 30-day rolling aggregates
"""

from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.window import Window
from bet.utils.spark_session import get_spark
from bet.utils.config import DataGenConfig
from bet.utils.logging_utils import get_logger
from bet.ingestion.last_activity_generator import generate_last_activity

logger = get_logger(__name__)


def prepare_num_data_inference(test_date: str) -> DataFrame:
    """
    Prepare feature vectors for churn inference on a specific date.
    
    Loads Silver-layer data and computes rolling window features from all transactions,
    sessions, and financial events up to (and including) the inference date. Ensures
    strict time causality - no future data leakage for production inference.
    
    Process:
    1. Load Silver layer tables (players, sessions, transactions, money_events)
    2. Extract numeric player indices from string player_id
    3. Filter all data to test_date or earlier
    4. Compute rolling features:
       - Session counts and net game results (7d, 30d)
       - Session duration and bet amount averages (30d)
       - Money event amounts (7d, 30d)
       - Balance snapshots at 7d and 30d ago
    5. Compute transaction features (failed withdrawals, deposits, withdrawals, ratio)
    6. Join all features with player snapshot for final inference data
    
    Args:
        test_date: Inference date in YYYY-MM-DD format
        
    Returns:
        DataFrame with player_idx and all computed features ready for model inference
        
    Raises:
        Exception: If Silver layer data is missing or corrupted
    """
    logger.info(f"Preparing inference data for date: {test_date}")
    
    spark = get_spark()
    spark.catalog.clearCache()
    
    # Load Silver layer data
    try:
        players_silver = spark.read.parquet("./data/silver/players")
        sessions_silver = spark.read.parquet("./data/silver/sessions")
        transactions_silver = spark.read.parquet("./data/silver/transactions")
        silver_money_events = spark.read.parquet("./data/silver/money_events")
        logger.info("Loaded all Silver layer tables successfully")
    except Exception as e:
        logger.error(f"Failed to load Silver layer data: {e}")
        raise

    # Extract numeric player indices
    players_silver = players_silver.drop('player_id')
    transactions_silver = transactions_silver.withColumn( "player_idx",
        F.regexp_replace("player_id", "[^0-9]", "").cast("long")).drop('player_id')
    sessions_silver = sessions_silver.withColumn( "player_idx",
        F.regexp_replace("player_id", "[^0-9]", "").cast("long")).drop('player_id')
    silver_money_events = silver_money_events.withColumn( "player_idx",
        F.regexp_replace("player_id", "[^0-9]", "").cast("long")).drop('player_id')

    # Filter data up to test_date
    logger.info(f"Filtering data up to {test_date}")
    silver_money_events = silver_money_events.filter( F.to_date("event_ts")<= F.lit(test_date)).withColumn('days_diff', F.datediff(F.lit(test_date), F.to_date('event_ts')))
    sessions_silver = sessions_silver.filter( F.to_date("session_date")<= F.lit(test_date)).withColumn('days_diff', F.datediff(F.lit(test_date), F.to_date('session_date')))
    transactions_silver = transactions_silver.filter( F.to_date("transaction_ts")<= F.lit(test_date)).withColumn('days_diff', F.datediff(F.lit(test_date),F.to_date(F.col('transaction_ts'))))

    first_last_activity = generate_last_activity(silver_money_events)
    player_snapshot = (players_silver
                    .select('player_idx','country','age_bucket','device_type',
                            'acquisition_channel', 'registration_date', 'risk_segment', 
                            'lifecycle_stage', F.col('current_balance'))
                    .join(first_last_activity,
                            on='player_idx',
                            how='left')
    )
    
    logger.info(f"Created player snapshot for {player_snapshot.count()} players")

    # Compute session features
    logger.info("Computing session features...")
    sessions_silver_one_date = (sessions_silver
    .filter((F.col('days_diff') < 30) & (F.col('days_diff') >=0))
    .groupBy('player_idx')
    .agg(
                F.sum(F.when(((F.col('days_diff') < 7) & (F.col('days_diff') >=0)), 1).otherwise(0)).alias('num_sessions_7d'),
                F.sum(F.when(((F.col('days_diff') < 7) & (F.col('days_diff') >=0)), F.col('signed_amount')).otherwise(0)).alias('net_game_result_7d'),
                F.count('*').alias('num_sessions_30d'),
                F.avg(F.col('session_duration_sec')).cast('int').alias('avg_sessions_duration_30d'),
                F.avg(F.col('total_bet_amount')).alias('avg_bet_amount_30d'),
                F.sum(F.col('signed_amount')).alias('net_game_result_30d'),
    )           
    )
    
    for c in sessions_silver_one_date.columns:
        assert sessions_silver_one_date.filter(F.col(c).isNull()).count() == 0
    logger.info(f"Session features computed for {sessions_silver_one_date.count()} players")

    # Compute money event features
    logger.info("Computing money event features...")
    silver_money_events_net = (silver_money_events
    .filter((F.col('days_diff') < 30) & (F.col('days_diff') >=0))
    .groupBy('player_idx')
    .agg(
                F.sum(F.when(((F.col('days_diff') < 7) & (F.col('days_diff') >=0)), F.col('signed_amount')).otherwise(0)).alias('net_amount_result_7d'),
                F.sum(F.col('signed_amount')).alias('net_amount_result_30d'),
    )        
    )

    for c in silver_money_events_net.columns:
        assert silver_money_events_net.filter(F.col(c).isNull()).count() == 0

    w = Window.partitionBy("player_idx").orderBy(F.col("event_ts").desc())

    player_30d = (silver_money_events
    .filter(F.col('days_diff') >=29)
    .withColumn('rn', F.row_number().over(w))
    .filter(F.col('rn') ==1)
    )

    player_30d_act= (players_silver
                    .select('player_idx','current_balance')
                    .join(player_30d.select('player_idx','balance_after_txn'),
                            on='player_idx',
                            how='left')
                                .withColumn(
                                "balance_30d_ago",
                                F.coalesce("balance_after_txn", "current_balance"))
                                .drop( 'current_balance', 'balance_after_txn')
    )

    for c in player_30d_act.columns:
        assert player_30d_act.filter(F.col(c).isNull()).count() == 0

    w = Window.partitionBy("player_idx").orderBy(F.col("event_ts").desc())
    player_7d = (silver_money_events
    .filter(F.col('days_diff') >=6)
    .withColumn('rn', F.row_number().over(w))
    .filter(F.col('rn') ==1)
    )
    player_7d_act= (players_silver
                    .select('player_idx','current_balance')
                    .join(player_7d.select('player_idx',F.col('balance_after_txn')),
                            on='player_idx',
                            how='left')
                                .withColumn(
                                "balance_7d_ago",
                                F.coalesce("balance_after_txn", "current_balance"))
                                .drop('current_balance', 'balance_after_txn')
    )

    for c in player_7d_act.columns:
        assert player_7d_act.filter(F.col(c).isNull()).count() == 0

    silver_money_events_one_date = (silver_money_events_net
    .join(player_30d_act, how='left', on='player_idx') 
    .join(player_7d_act, how='left', on='player_idx')
    )

    for c in silver_money_events_one_date.columns:
        assert silver_money_events_one_date.filter(F.col(c).isNull()).count() == 0
    logger.info(f"Money event features computed for {silver_money_events_one_date.count()} players")

    # Compute transaction features
    logger.info("Computing transaction features...")
    transactions_silver_one_date = (transactions_silver
    .filter(F.col('days_diff')<30)
    .groupBy(F.col('player_idx'))
    .agg(
        F.sum(F.when((F.col('transaction_type')=='withdrawal') & (F.col('success_flag')==False),1).otherwise(0)).alias('failed_withdrawals_30d'),
        F.sum(F.when(F.col('transaction_type')=='deposit',1).otherwise(0)).alias('deposit_count_30d'),
        F.sum(F.when(F.col('transaction_type')=='withdrawal',1).otherwise(0)).alias('withdrawal_count_30d'),
    )
    .withColumn( 'withdrawal_ratio',
        F.when(
                F.col("deposit_count_30d") > 0,
                F.col("withdrawal_count_30d") / F.col("deposit_count_30d")
            ).otherwise(F.lit(0.0)) ) )

    for c in transactions_silver_one_date.columns:
        assert transactions_silver_one_date.filter(F.col(c).isNull()).count() == 0
    logger.info(f"Transaction features computed for {transactions_silver_one_date.count()} players")
    
    # Combine all features
    gold_player_behavior = (player_snapshot
                .filter(F.col('first_session_date').isNotNull())
                .filter(F.datediff(F.lit(test_date), F.col("last_session_date")) < 7)
                .filter(F.datediff(F.lit(test_date), F.col("first_session_date")) > 30)
                .select('player_idx')
                .join(silver_money_events_one_date, how='inner', on='player_idx') 
                .join(transactions_silver_one_date, how='left', on='player_idx') 
                .join(sessions_silver_one_date, how='left', on='player_idx')
    )

    # Fill nulls
    for c in gold_player_behavior.columns:
        if gold_player_behavior.filter(F.col(c).isNull()).count() != 0:
            logger.debug(f"Filling null values in column: {c}")

    gold_player_behavior = gold_player_behavior.fillna(0)
    logger.info(f"Inference data prepared for {gold_player_behavior.count()} players on {test_date}")
    return gold_player_behavior


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prepare_data_inference.py <test_date>")
        print("  Example: python prepare_data_inference.py 2024-03-15")
        sys.exit(1)
    
    test_date = sys.argv[1]
    logger.info(f"Running inference data preparation for {test_date}")
    result = prepare_num_data_inference(test_date)
    logger.info(f"Successfully prepared {result.count()} player inference records")















