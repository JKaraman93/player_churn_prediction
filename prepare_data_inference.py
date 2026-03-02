
from src.utils.spark_session import get_spark
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd
from pyspark.sql.functions import pandas_udf
import src.utils.config as config
from src.ingestion.last_activity_generator import generate_last_activity



def prepare_data_inference(test_date):
    spark = get_spark()
    spark.catalog.clearCache()
    players_silver = spark.read.parquet("./data/silver/players")
    sessions_silver = spark.read.parquet("./data/silver/sessions")
    transactions_silver = spark.read.parquet("./data/silver/transactions")
    silver_money_events = spark.read.parquet("./data/silver/money_events")

    players_silver = players_silver.drop('player_id')
    transactions_silver = transactions_silver.withColumn( "player_idx",
        F.regexp_replace("player_id", "[^0-9]", "").cast("long")).drop('player_id')
    sessions_silver = sessions_silver.withColumn( "player_idx",
        F.regexp_replace("player_id", "[^0-9]", "").cast("long")).drop('player_id')
    silver_money_events = silver_money_events.withColumn( "player_idx",
        F.regexp_replace("player_id", "[^0-9]", "").cast("long")).drop('player_id')


    ## dates up to test date
    silver_money_events = silver_money_events.filter( F.to_date("event_ts")<= F.lit(test_date)).withColumn('days_diff', F.date_diff(F.lit(test_date), F.to_date('event_ts')))
    sessions_silver = sessions_silver.filter( F.to_date("session_date")<= F.lit(test_date)).withColumn('days_diff', F.date_diff(F.lit(test_date), F.to_date('session_date')))
    transactions_silver = transactions_silver.filter( F.to_date("transaction_ts")<= F.lit(test_date)).withColumn('days_diff', F.date_diff(F.lit(test_date),F.to_date(F.col('transaction_ts'))))


    first_last_activity = generate_last_activity(silver_money_events)
    player_snapshot = (players_silver
                    .select('player_idx','country','age_bucket','device_type',
                            'acquisition_channel', 'registration_date', 'risk_segment', 
                            'lifecycle_stage', F.col('current_balance'))
                    .join(first_last_activity,
                            on='player_idx',
                            how='left')
    )

    ## Define the date, you want to inference ## 
    #test_date = config_.end_date

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

    sessions_silver_one_date.persist()
    sessions_silver_one_date.count() 
    
    for c in sessions_silver_one_date.columns:
        assert sessions_silver_one_date.filter(F.col(c).isNull()).count() == 0


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

    silver_money_events_one_date.persist()
    silver_money_events_one_date.count() 

    for c in silver_money_events_one_date.columns:
        assert silver_money_events_one_date.filter(F.col(c).isNull()).count() == 0


    transactions_silver_one_date = (transactions_silver
    .filter(F.col('days_diff')<30)
    .groupBy(F.col('player_idx'))
    .agg(
        F.sum(F.when((F.col('transaction_type')=='withdrawal') & (F.col('success_flag')==False),1).otherwise(0)).alias('failed_withdrawals_30d'),
        F.sum(F.when(F.col('transaction_type')=='deposit',1).otherwise(0)).alias('deposit_count_30d'),
        F.sum(F.when(F.col('transaction_type')=='withdrawal',1).otherwise(0)).alias('withdrawal_count_30d'),
    )
    .withColumn(        'withdrawal_ratio',
        F.when(
                F.col("deposit_count_30d") > 0,
                F.col("withdrawal_count_30d") / F.col("deposit_count_30d")
            ).otherwise(F.lit(0.0))
    )
    )

    for c in transactions_silver_one_date.columns:
        assert transactions_silver_one_date.filter(F.col(c).isNull()).count() == 0
    


    gold_player_behavior = (player_snapshot
                .filter(F.col('first_session_date').isNotNull())  # exclude new players
                #.filter(F.datediff(F.lit(test_date), F.col("last_session_date")) < 7)    # if someone hasn't a session in the last 7 days, he has already churned      
                .select('player_idx')
                .join(silver_money_events_one_date, how='inner', on='player_idx') 
                .join(transactions_silver_one_date, how='left', on='player_idx') 
                .join(sessions_silver_one_date, how='left', on='player_idx') 

    )

    gold_player_behavior.persist()
    gold_player_behavior.count() 


    ## only transaction-related features must has null values
    for c in gold_player_behavior.columns:
        if gold_player_behavior.filter(F.col(c).isNull()).count() != 0    :
            print(c) 




    gold_player_behavior = gold_player_behavior.fillna(0)
    return gold_player_behavior















