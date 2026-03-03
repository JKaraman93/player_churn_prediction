from src.utils.spark_session import get_spark
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd
from pyspark.sql.functions import pandas_udf
import src.utils.config as config
from src.ingestion.last_activity_generator import generate_last_activity


spark = get_spark()
spark.catalog.clearCache()
config_ = config.DataGenConfig()
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

first_last_activity = generate_last_activity(silver_money_events)
player_snapshot = (players_silver
                   .select('player_idx','country','age_bucket','device_type',
                           'acquisition_channel', 'registration_date', 'risk_segment', 
                           'lifecycle_stage', F.col('current_balance'))
                   .join(first_last_activity,
                         on='player_idx',
                         how='left')
)


# ## Construct training dataset for ML

# ### Full dataframe (players x dates)

date_range = spark.sql(
    f"SELECT explode(sequence(to_date('{config_.start_date}'), to_date('{config_.end_date}'), interval 1 day)) AS reference_date"
)

df_pl_date = (player_snapshot
              .select('player_idx', 'registration_date','first_event_date', 'last_event_date',  'first_session_date',  'last_session_date', 'first_financial_date', 'last_financial_date')
              .crossJoin(date_range)
            .filter(F.col('first_session_date').isNotNull())  # exclude new players
            .withColumn('session_date_days', 
                F.datediff(F.col("reference_date"), F.lit("1970-01-01")))
)
window_7d = (Window.partitionBy('player_idx').orderBy('session_date_days').rangeBetween(-6,0))
window_30d = (Window.partitionBy('player_idx').orderBy('session_date_days').rangeBetween(-29,0))

## number of days of pereiod examined is : 182 (3 months)
assert players_silver.filter(F.col('lifecycle_stage')=='new').count()*182 + df_pl_date.count() == player_snapshot.count()*182


# ### Compute rolling futures based on sessions
df_sessions_detail = (df_pl_date.select('player_idx','reference_date','registration_date', 'first_session_date','last_session_date','session_date_days')
.join(sessions_silver
      .select('session_id',
 'session_duration_sec',
 'bet_count',
 'total_bet_amount',
 'total_win_amount',
 'signed_amount',
 'balance_after_txn',
   F.to_date('session_date').alias('reference_date'), 
   'player_idx'),
        how='left', on=['player_idx','reference_date'])
.filter( F.col("first_session_date")<= F.col("reference_date"))
#.filter(F.datediff(F.col("reference_date"), F.col("first_session_date")) > 30)
#.filter(F.datediff(F.lit(config_.end_date), F.col("reference_date")) > 7)
.orderBy('player_idx', 'reference_date' )
)


df_sessions_rolling= (df_sessions_detail
            .withColumn('num_sessions_7d', F.count('session_id').over(window_7d))
            .withColumn('net_game_result_7d',F.coalesce(F.sum('signed_amount').over(window_7d), F.lit(0)))
            .withColumn('num_sessions_30d', F.count('session_id').over(window_30d))
            .withColumn('net_game_result_30d', F.coalesce(F.sum('signed_amount').over(window_30d), F.lit(0))) 
            .withColumn('avg_sessions_duration_30d', F.coalesce(F.avg('session_duration_sec').over(window_30d), F.lit(0)))
            .withColumn('avg_bet_amount_30d', F.coalesce(F.avg('total_bet_amount').over(window_30d), F.lit(0)))
                       ) 
df_sessions_rolling = df_sessions_rolling.select(
                            'player_idx',
                            'reference_date',
                            'num_sessions_7d',
                            'net_game_result_7d',
                            'num_sessions_30d',
                            'avg_sessions_duration_30d',
                            'avg_bet_amount_30d',
                            'net_game_result_30d',
                            'first_session_date').drop_duplicates()


df_sessions_rolling.persist()
df_sessions_rolling.count() 

for c in df_sessions_rolling.columns:
   assert df_sessions_rolling.filter(F.col(c).isNull()).count() == 0



# ### Compute rolling futures based on all events


silver_money_events_detail =  (df_pl_date.select('player_idx','reference_date','registration_date', 'first_event_date','last_event_date','session_date_days')
.join(silver_money_events
      .select('event_id',
 'event_type',
 'signed_amount',
 'balance_after_txn',
 'event_ts',
   F.to_date(F.col('event_ts')).alias('reference_date'), 
   'player_idx'),
        how='left', on=['player_idx','reference_date'])
.filter( F.col("first_event_date")<= F.col("reference_date"))
.orderBy('player_idx', 'reference_date' )
)


window_up_to_7d = (Window.partitionBy('player_idx').orderBy('session_date_days').rangeBetween(Window.unboundedPreceding,-6))
window_up_to_30d = (Window.partitionBy('player_idx').orderBy('session_date_days').rangeBetween(Window.unboundedPreceding,-29))

silver_money_events_rolling = (silver_money_events_detail
            .withColumn('net_amount_result_7d', F.coalesce( F.sum('signed_amount').over(window_7d), F.lit(0)))
            .withColumn('net_amount_result_30d',F.coalesce( F.sum('signed_amount').over(window_30d), F.lit(0))))

window_latest_day_session = (Window.partitionBy('player_idx','reference_date').orderBy(F.col("event_ts").desc()))

silver_money_events_rolling = ( silver_money_events_rolling
    .withColumn("rn", F.row_number().over(window_latest_day_session))
    .filter(F.col("rn") == 1)
    .drop("rn")
)

silver_money_events_rolling = (silver_money_events_rolling
            .withColumn('balance_7d_ago', F.last('balance_after_txn', ignorenulls=True).over(window_up_to_7d))
            .withColumn('balance_30d_ago', F.last('balance_after_txn', ignorenulls=True).over(window_up_to_30d)))

silver_money_events_rolling = (silver_money_events_rolling
.filter(F.datediff(F.col("reference_date"), F.col("first_event_date")) > 30)
.filter(F.datediff(F.lit(config_.end_date), F.col("reference_date")) > 7)
.select(
                            'player_idx',
                            'reference_date',
                            'balance_7d_ago',
                            'balance_30d_ago',
                            'net_amount_result_7d',
                            'net_amount_result_30d').drop_duplicates())



silver_money_events_rolling.persist()
silver_money_events_rolling.count() 

for c in silver_money_events_rolling.columns:
   assert silver_money_events_rolling.filter(F.col(c).isNull()).count() == 0



# ### Compute rolling futures based on financial transactions 


transactions_detail = (df_pl_date.select('player_idx','reference_date','registration_date', 'first_financial_date','last_financial_date','session_date_days')
.join(transactions_silver
      .select('transaction_id',
 'transaction_type',
 'amount',
 'success_flag',
 'signed_amount',
 'balance_after_txn',
   F.to_date(F.col('transaction_ts')).alias('reference_date'), 
   'player_idx'),
        how='left', on=['player_idx','reference_date'])
.filter( F.col("first_financial_date")<= F.col("reference_date"))
#.filter(F.datediff(F.col("reference_date"), F.col("first_session_date")) > 30)
#.filter(F.datediff(F.lit(config_.end_date), F.col("reference_date")) > 7)
.orderBy('player_idx', 'reference_date' )
)




transactions_rolling = (transactions_detail
                .withColumn('failed_withdrawals_30d', 
                    F.sum(F.when((F.col('success_flag')==False) & (F.col('transaction_type')=='withdrawal'), 1).otherwise(0)).over(window_30d))
                .withColumn('deposit_count_30d', 
                    F.sum(F.when(F.col('transaction_type')=='deposit',1).otherwise(0)).over(window_30d))
                .withColumn('withdrawal_count_30d', 
                    F.sum(F.when(F.col('transaction_type')=='withdrawal',1).otherwise(0)).over(window_30d))
                .withColumn('withdrawal_ratio',
                     F.when(
                        F.col("deposit_count_30d") > 0,
                        F.col("withdrawal_count_30d") / F.col("deposit_count_30d")
                         ).otherwise(F.lit(0.0)))
)

transactions_rolling = transactions_rolling.select(
                            'player_idx',
                            'reference_date',
                            'failed_withdrawals_30d',
                            'deposit_count_30d',
                            'withdrawal_count_30d',
                            'withdrawal_ratio',
).drop_duplicates()


transactions_rolling.persist()
transactions_rolling.count() 

for c in df_sessions_rolling.columns:
   assert df_sessions_rolling.filter(F.col(c).isNull()).count() == 0



# ### Compose the final training dataset


gold_player_behavior = (silver_money_events_rolling
                        .join(df_sessions_rolling,
                        how='left', on=['player_idx', 'reference_date'])
                        .join(transactions_rolling,
 how='left', on=['player_idx', 'reference_date']) 
)

silver_money_events_rolling.unpersist()
df_sessions_rolling.unpersist()
transactions_rolling.unpersist()



gold_player_behavior.persist()
gold_player_behavior.count()



## keep only days that follow the first session date of each player since 
gold_player_behavior = (gold_player_behavior
.filter(F.datediff(F.col("reference_date"), F.col("first_session_date")) > 30)     
.filter(F.datediff(F.lit(config_.end_date), F.col("reference_date")) > 7)
.drop(F.col('first_session_date')) )


## in case of sessions or transactions dates, one  follow the other -> null values 
for c in gold_player_behavior.columns[2:]:  # exclude player_idx and reference_date 
    gold_player_behavior = gold_player_behavior.withColumn(
        c, F.coalesce(F.col(c), F.lit(0))
    )


for c in gold_player_behavior.columns:
    assert gold_player_behavior.filter(F.col(c).isNull()).count() == 0
gold_player_behavior.count()
gold_player_behavior.unpersist()


# ### Create gold labels



df_num_of_sessions = (sessions_silver
    .withColumn('session_date', F.to_date('session_date'))  # Convert to date and assign a proper column name
    .groupBy('player_idx', 'session_date')  # Group by player_idx and the converted session_date
    .agg(F.count('*').alias('num_of_session')))

df_sessions = (df_pl_date
.join(df_num_of_sessions
      .select('num_of_session', F.col('session_date').alias('reference_date'), 'player_idx'),
        how='left', on=['player_idx','reference_date'])
)

sessions_silver_sec = (df_sessions
                       .filter(F.col('reference_date') >= F.col('first_session_date'))
            .withColumn('num_sessions_7d', 
                        F.sum(F.when(F.col("num_of_session") > 0, 1).otherwise(0)).over(window_7d))
            .withColumn('churn_7d', F.when(F.col('num_sessions_7d')==0, True).otherwise(False))
            .withColumn('num_sessions_30d', 
                        F.sum(F.when(F.col('num_of_session') > 0, 1).otherwise(0)).over(window_30d))
           .withColumn('churn_30d', F.when(F.col('num_sessions_30d')==0, True).otherwise(False))
)


window_7d_ahead = (Window.partitionBy('player_idx').orderBy('session_date_days').rangeBetween(1,6))
target = (sessions_silver_sec
            .withColumn('next_7d_churn', 
            (F.max(F.col("churn_7d").cast("int")).over(window_7d_ahead) == 1))
)


gold_labels = (target
.select('player_idx', 'reference_date','next_7d_churn')
.filter(F.datediff(F.col("reference_date"), F.col("first_session_date")) > 30)
.filter(F.datediff(F.lit(config_.end_date), F.col("reference_date")) > 7))


assert gold_labels.filter(F.col("next_7d_churn").isNull()).count() == 0
assert gold_player_behavior.count() == gold_labels.count()



player_snapshot.write.mode("overwrite").parquet("./data/gold/player_snapshot")
gold_player_behavior.write.mode("overwrite").parquet("./data/gold/player_behavior")
gold_labels.write.mode("overwrite").parquet("./data/gold/labels")


