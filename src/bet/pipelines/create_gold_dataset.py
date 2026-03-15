"""
Gold Layer Generation: ML-Ready Features

This module generates the Gold layer datasets used for training and evaluation.
It transforms Silver-layer data into ML-ready feature tables with rolling window
aggregations to capture temporal patterns in player behavior.

Generated tables:

1. player_snapshot: Static player attributes
   - Demographics (country, age bucket, acquisition channel)
   - Lifecycle and risk segments
   - Registration date and current balance
   - Last activity metrics

2. player_behavior: Rolling behavioral features
   - 7-day and 30-day aggregates of betting and session activity
   - Financial metrics (net profit, deposits, withdrawals)
   - Session statistics (count, duration, frequency)
   - Historical balance snapshots at different time windows

3. labels: Churn labels for supervised learning
   - Binary target indicating 7-day inactivity completion
   - Aligned with player_behavior for training

Key design principles:
- Strict time causality (features use only data <= reference_date)
- Zero-activity preservation (inactive players retained with zeros)
- Rolling window consistency for production inference

Inputs:
- Silver layer parquet tables (players, sessions, transactions, money_events)

Outputs:
- Parquet tables in data/gold/ directory
"""

from bet.utils.spark_session import get_spark
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window
from bet.utils.config import DataGenConfig
from bet.utils.logging_utils import get_logger
from bet.ingestion.last_activity_generator import generate_last_activity

logger = get_logger(__name__)


def _create_session_features(df_pl_date: DataFrame, sessions_silver: DataFrame, window_7d: Window, window_30d: Window) -> DataFrame:
    """
    Create rolling window session features: counts and net game results.
    
    Args:
        df_pl_date: Player x date cross join
        sessions_silver: Silver sessions data
        window_7d: 7-day window specification
        window_30d: 30-day window specification
        
    Returns:
        DataFrame with session rolling features
    """
    df_sessions_detail = (df_pl_date.select('player_idx','reference_date','registration_date', 'first_session_date','last_session_date','session_date_days')
        .join(sessions_silver
            .select('session_id', 'session_duration_sec', 'bet_count', 'total_bet_amount', 'total_win_amount', 'signed_amount', 'balance_after_txn',
                F.to_date('session_date').alias('reference_date'), 'player_idx'),
            how='left', on=['player_idx','reference_date'])
        .filter(F.col("first_session_date") <= F.col("reference_date"))
        .orderBy('player_idx', 'reference_date')
    )
    
    df_sessions_rolling = (df_sessions_detail
        .withColumn('num_sessions_7d', F.count('session_id').over(window_7d))
        .withColumn('net_game_result_7d', F.coalesce(F.sum('signed_amount').over(window_7d), F.lit(0)))
        .withColumn('num_sessions_30d', F.count('session_id').over(window_30d))
        .withColumn('net_game_result_30d', F.coalesce(F.sum('signed_amount').over(window_30d), F.lit(0)))
        .withColumn('avg_sessions_duration_30d', F.coalesce(F.avg('session_duration_sec').over(window_30d), F.lit(0)))
        .withColumn('avg_bet_amount_30d', F.coalesce(F.avg('total_bet_amount').over(window_30d), F.lit(0)))
    )
    
    result = df_sessions_rolling.select(
        'player_idx', 'reference_date', 'num_sessions_7d', 'net_game_result_7d', 'num_sessions_30d', 
        'avg_sessions_duration_30d', 'avg_bet_amount_30d', 'net_game_result_30d', 'first_session_date'
    ).drop_duplicates()
    
    return result


def _create_money_event_features(df_pl_date: DataFrame, silver_money_events: DataFrame, window_7d: Window, window_30d: Window, 
                                 window_up_to_7d: Window, window_up_to_30d: Window, config: DataGenConfig) -> DataFrame:
    """
    Create rolling window money event features: balances and net amounts.
    
    Args:
        df_pl_date: Player x date cross join
        silver_money_events: Silver money events data
        window_7d: 7-day window
        window_30d: 30-day window
        window_up_to_7d: Up to 7 days window
        window_up_to_30d: Up to 30 days window
        config: DataGenConfig instance
        
    Returns:
        DataFrame with money event rolling features
    """
    money_events_detail = (df_pl_date.select('player_idx','reference_date','registration_date', 'first_event_date','last_event_date','session_date_days')
        .join(silver_money_events
            .select('event_id', 'event_type', 'signed_amount', 'balance_after_txn', 'event_ts',
                F.to_date(F.col('event_ts')).alias('reference_date'), 'player_idx'),
            how='left', on=['player_idx','reference_date'])
        .filter(F.col("first_event_date") <= F.col("reference_date"))
        .orderBy('player_idx', 'reference_date')
    )
    
    money_events_rolling = (money_events_detail
        .withColumn('net_amount_result_7d', F.coalesce(F.sum('signed_amount').over(window_7d), F.lit(0)))
        .withColumn('net_amount_result_30d', F.coalesce(F.sum('signed_amount').over(window_30d), F.lit(0)))
    )
    
    window_latest_day = (Window.partitionBy('player_idx','reference_date').orderBy(F.col("event_ts").desc()))
    money_events_rolling = (money_events_rolling
        .withColumn("rn", F.row_number().over(window_latest_day))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )
    
    money_events_rolling = (money_events_rolling
        .withColumn('balance_7d_ago', F.last('balance_after_txn', ignorenulls=True).over(window_up_to_7d))
        .withColumn('balance_30d_ago', F.last('balance_after_txn', ignorenulls=True).over(window_up_to_30d))
    )
    
    result = (money_events_rolling
        .filter(F.datediff(F.col("reference_date"), F.col("first_event_date")) > 30)
        .filter(F.datediff(F.lit(config.end_date), F.col("reference_date")) > 7)
        .select('player_idx', 'reference_date', 'balance_7d_ago', 'balance_30d_ago', 'net_amount_result_7d', 'net_amount_result_30d')
        .drop_duplicates()
    )
    
    return result


def _create_transaction_features(df_pl_date: DataFrame, transactions_silver: DataFrame, window_30d: Window) -> DataFrame:
    """
    Create rolling window transaction features: failure counts and withdrawal ratios.
    
    Args:
        df_pl_date: Player x date cross join
        transactions_silver: Silver transactions data
        window_30d: 30-day window
        
    Returns:
        DataFrame with transaction rolling features
    """
    transactions_detail = (df_pl_date.select('player_idx','reference_date','registration_date', 'first_financial_date','last_financial_date','session_date_days')
        .join(transactions_silver
            .select('transaction_id', 'transaction_type', 'amount', 'success_flag', 'signed_amount', 'balance_after_txn',
                F.to_date(F.col('transaction_ts')).alias('reference_date'), 'player_idx'),
            how='left', on=['player_idx','reference_date'])
        .filter(F.col("first_financial_date") <= F.col("reference_date"))
        .orderBy('player_idx', 'reference_date')
    )
    
    transactions_rolling = (transactions_detail
        .withColumn('failed_withdrawals_30d', F.sum(F.when((F.col('success_flag')==False) & (F.col('transaction_type')=='withdrawal'), 1).otherwise(0)).over(window_30d))
        .withColumn('deposit_count_30d', F.sum(F.when(F.col('transaction_type')=='deposit',1).otherwise(0)).over(window_30d))
        .withColumn('withdrawal_count_30d', F.sum(F.when(F.col('transaction_type')=='withdrawal',1).otherwise(0)).over(window_30d))
        .withColumn('withdrawal_ratio', F.when(F.col("deposit_count_30d") > 0, F.col("withdrawal_count_30d") / F.col("deposit_count_30d")).otherwise(F.lit(0.0)))
    )
    
    result = transactions_rolling.select(
        'player_idx', 'reference_date', 'failed_withdrawals_30d', 'deposit_count_30d', 'withdrawal_count_30d', 'withdrawal_ratio'
    ).drop_duplicates()
    
    return result


def main() -> None:
    """
    Generate Gold layer with ML-ready features.
    
    Transforms Silver layer data into three Gold layer tables:
    1. player_snapshot: Static player attributes with last activity dates
    2. player_behavior: Rolling window behavioral features (7d and 30d windows)
    3. labels: Churn labels (7-day inactivity binary target)
    
    Key design:
    - Enforces time causality (features use only data <= reference_date)
    - Preserves zero-activity records with coalesce to 0
    - Excludes new players and future prediction periods
    
    Outputs to data/gold/ directory.
    
    Returns:
        None
    """
    logger.info("Starting Gold layer generation")
    
    spark = get_spark()
    spark.catalog.clearCache()
    
    config = DataGenConfig()
    
    # Load Silver data
    logger.info("Reading Silver layer data")
    players_silver = spark.read.parquet("./data/silver/players")
    sessions_silver = spark.read.parquet("./data/silver/sessions")
    transactions_silver = spark.read.parquet("./data/silver/transactions")
    silver_money_events = spark.read.parquet("./data/silver/money_events")
    
    # Extract player_idx and drop player_id
    logger.info("Extracting player indices")
    players_silver = players_silver.drop('player_id')
    transactions_silver = transactions_silver.withColumn("player_idx", F.regexp_replace("player_id", "[^0-9]", "").cast("long")).drop('player_id')
    sessions_silver = sessions_silver.withColumn("player_idx", F.regexp_replace("player_id", "[^0-9]", "").cast("long")).drop('player_id')
    silver_money_events = silver_money_events.withColumn("player_idx", F.regexp_replace("player_id", "[^0-9]", "").cast("long")).drop('player_id')
    
    # Create player snapshot
    logger.info("Creating player snapshot")
    first_last_activity = generate_last_activity(silver_money_events)
    player_snapshot = (players_silver
        .select('player_idx','country','age_bucket','device_type', 'acquisition_channel', 'registration_date', 'risk_segment', 'lifecycle_stage', F.col('current_balance'))
        .join(first_last_activity, on='player_idx', how='left')
    )
    
    # Create player x date cross join
    logger.info("Creating player x date cross join")
    date_range = spark.sql(f"SELECT explode(sequence(to_date('{config.start_date}'), to_date('{config.end_date}'), interval 1 day)) AS reference_date")
    
    df_pl_date = (player_snapshot
        .select('player_idx', 'registration_date','first_event_date', 'last_event_date', 'first_session_date', 'last_session_date', 'first_financial_date', 'last_financial_date')
        .crossJoin(date_range)
        .filter(F.col('first_session_date').isNotNull())
        .withColumn('session_date_days', F.datediff(F.col("reference_date"), F.lit("1970-01-01")))
    )
    
    logger.info(f"Created {df_pl_date.count()} player-date combinations")
    
    # Define windows
    window_7d = (Window.partitionBy('player_idx').orderBy('session_date_days').rangeBetween(-6,0))
    window_30d = (Window.partitionBy('player_idx').orderBy('session_date_days').rangeBetween(-29,0))
    window_up_to_7d = (Window.partitionBy('player_idx').orderBy('session_date_days').rangeBetween(Window.unboundedPreceding,-6))
    window_up_to_30d = (Window.partitionBy('player_idx').orderBy('session_date_days').rangeBetween(Window.unboundedPreceding,-29))
    
    # Create feature sets
    logger.info("Computing session features")
    df_sessions_rolling = _create_session_features(df_pl_date, sessions_silver, window_7d, window_30d)
    
    logger.info("Computing money event features")
    silver_money_events_rolling = _create_money_event_features(df_pl_date, silver_money_events, window_7d, window_30d, window_up_to_7d, window_up_to_30d, config)
    
    logger.info("Computing transaction features")
    transactions_rolling = _create_transaction_features(df_pl_date, transactions_silver, window_30d)
    
    # Combine behavior features
    logger.info("Composing final behavior dataset")
    gold_player_behavior = (silver_money_events_rolling
        .join(df_sessions_rolling, how='left', on=['player_idx', 'reference_date'])
        .join(transactions_rolling, how='left', on=['player_idx', 'reference_date'])
    )
    
    # Filter and fill nulls
    gold_player_behavior = (gold_player_behavior
        .filter(F.datediff(F.col("reference_date"), F.col("first_session_date")) > 30)
        .filter(F.datediff(F.lit(config.end_date), F.col("reference_date")) > 7)
        .drop(F.col('first_session_date'))
    )
    
    for col in gold_player_behavior.columns[2:]:  # Exclude player_idx and reference_date
        gold_player_behavior = gold_player_behavior.withColumn(col, F.coalesce(F.col(col), F.lit(0)))
    
    logger.info(f"Created behavior features for {gold_player_behavior.count()} records")
    
    # Create labels
    logger.info("Creating churn labels")
    df_num_of_sessions = (sessions_silver
        .withColumn('session_date', F.to_date('session_date'))
        .groupBy('player_idx', 'session_date')
        .agg(F.count('*').alias('num_of_session'))
    )
    
    df_sessions = (df_pl_date
        .join(df_num_of_sessions
            .select('num_of_session', F.col('session_date').alias('reference_date'), 'player_idx'),
            how='left', on=['player_idx','reference_date'])
    )
    
    sessions_with_churn = (df_sessions
        .filter(F.col('reference_date') >= F.col('first_session_date'))
        .withColumn('num_sessions_7d', F.sum(F.when(F.col("num_of_session") > 0, 1).otherwise(0)).over(window_7d))
        .withColumn('churn_7d', F.when(F.col('num_sessions_7d')==0, True).otherwise(False))
        .withColumn('num_sessions_30d', F.sum(F.when(F.col('num_of_session') > 0, 1).otherwise(0)).over(window_30d))
        .withColumn('churn_30d', F.when(F.col('num_sessions_30d')==0, True).otherwise(False))
    )
    
    window_7d_ahead = (Window.partitionBy('player_idx').orderBy('session_date_days').rangeBetween(1,6))
    target = (sessions_with_churn
        .withColumn('next_7d_churn', F.max(F.col("churn_7d").cast("int")).over(window_7d_ahead) == 1)
    )
    
    gold_labels = (target
        .select('player_idx', 'reference_date','next_7d_churn')
        .filter(F.datediff(F.col("reference_date"), F.col("first_session_date")) > 30)
        .filter(F.datediff(F.lit(config.end_date), F.col("reference_date")) > 7)
    )
    
    logger.info(f"Created {gold_labels.count()} labels")
    
    # Validate alignment
    behavior_count = gold_player_behavior.count()
    labels_count = gold_labels.count()
    assert behavior_count == labels_count, f"Behavior ({behavior_count}) and labels ({labels_count}) counts must match"
    
    # Write to Gold layer
    logger.info("Writing tables to data/gold/")
    player_snapshot.write.mode("overwrite").parquet("./data/gold/player_snapshot")
    gold_player_behavior.write.mode("overwrite").parquet("./data/gold/player_behavior")
    gold_labels.write.mode("overwrite").parquet("./data/gold/labels")
    
    logger.info("Gold layer generation completed successfully")


if __name__ == "__main__":
    main()


