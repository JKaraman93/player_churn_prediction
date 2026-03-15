"""
Silver Layer Generation: Data Cleaning and Standardization

This module transforms raw Bronze layer data into cleaned, time-consistent
Silver layer tables suitable for feature engineering.

Transformations applied:
1. Remove duplicates and handle missing values
2. Standardize data types and formats
3. Enforce business rules (e.g., lifecycle stage filters)
4. Create consistent player identifiers
5. Aggregate transaction records and money events

Generated tables:
- players: Deduplicated player profiles with balanced amounts
- sessions: Unique player-session combinations
- transactions: Deduplicated financial transactions
- money_events: Aggregated cash flow events

Outputs:
- Parquet tables in data/silver/ directory
- Source of truth for all downstream feature engineering
"""

from bet.utils.spark_session import get_spark
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd

spark = get_spark()
spark.catalog.clearCache()
players_bronze = spark.read.parquet("./data/bronze/players")
sessions_bronze = spark.read.parquet("./data/bronze/sessions")
transactions_bronze = spark.read.parquet("./data/bronze/transactions")

sessions_bronze = sessions_bronze.dropDuplicates(['player_id', 'session_date'])

silver_players = (
    players_bronze
    .dropDuplicates(["player_id"])
    .withColumn(
        "balance",
        F.coalesce(F.col("balance"), F.lit(0.0))
    )
    .filter(F.col("lifecycle_stage").isin("new", "engaged", "at_risk", "churned"))
)

silver_sessions = (
    sessions_bronze
    .join(
        silver_players.select("player_id","balance","registration_date"),
        on="player_id",
        how="inner"
    )
    .filter(F.col("session_duration_sec") >= 0)
    .filter(F.col('session_date') >= F.col('registration_date'))
    .drop(F.col('registration_date'))
    .withColumn("bet_count", F.coalesce(F.col("bet_count"), F.lit(0)))
    .withColumn("total_bet_amount", F.coalesce(F.col("total_bet_amount"), F.lit(0.0)))
    .withColumn("total_win_amount", F.coalesce(F.col("total_win_amount"), F.lit(0.0)))
    .withColumn("signed_amount",
    F.when(F.col("total_win_amount") == 0, -F.col("total_bet_amount"))
        .otherwise(F.col("total_win_amount"))
    )
)


silver_transactions = (
    transactions_bronze
    .withColumn(
        "signed_amount", 
        F.when(F.col("transaction_type") == "deposit", F.col("amount") * F.col("success_flag").cast('int'))
         .when(F.col("transaction_type") == "withdrawal", -F.col("amount") * F.col("success_flag").cast('int'))
         .otherwise(F.lit(0.0))

    )
)

silver_transactions = (
    silver_transactions
    .join(
        silver_players.select("player_id", "balance",'registration_date'),
        on="player_id",
        how="inner"
    )
        .filter(F.col('transaction_ts') >= F.col('registration_date'))
        .drop(F.col('registration_date'))

)

all_events = ( silver_sessions
                .withColumn('event_type', F.lit('session'))
                .select('player_id', 
                    F.col('session_id').alias('event_id'),   
                    F.col('session_date').alias('event_ts'), 
                'event_type',
                'signed_amount',
                'balance') 
                .unionByName(silver_transactions
                .select('player_id', 
                    F.col('transaction_id').alias('event_id'),   
                    F.col('transaction_ts').alias('event_ts'), 
                    F.col('transaction_type').alias('event_type'),
                'signed_amount',
                'balance'))

)

def process_transactions(pdf):
    pdf = pdf.sort_values("event_ts")
    balance = pdf["balance"].iloc[0]
    balance_after_txn_list = []
    rows_to_delete = []

    for i, row in pdf.iterrows():
        balance_after_txn = balance + row["signed_amount"]
        if balance_after_txn > 0:
            balance_after_txn_list.append(balance_after_txn)
            balance = balance_after_txn
        else:
            rows_to_delete.append(i)
            balance_after_txn_list.append(balance)
            
    pdf["balance"] = balance_after_txn_list
    return pdf.drop(index=rows_to_delete).reset_index(drop=True)

# 3. Apply the function using applyInPandas with the new_schema
silver_money_events = (all_events
.groupBy("player_id").applyInPandas(
    process_transactions, 
    schema=all_events.schema
)
.withColumnRenamed('balance', 'balance_after_txn')
)

silver_transactions_final = (silver_transactions
                       .join(silver_money_events.select(F.col('event_id').alias('transaction_id'),'balance_after_txn'),
                             how='inner',
                             on='transaction_id')
                            .drop('balance')
)

silver_sessions_final = (silver_sessions
                       .join(silver_money_events.select(F.col('event_id').alias('session_id'),'balance_after_txn'),
                             how='inner',
                             on='session_id')
                           .drop('balance',) 
)

player_window = (Window
    .partitionBy("player_id")
    .orderBy("event_ts")
    #.rowsBetween(Window.unboundedPreceding, Window.currentRow)
)
silver_players_final = (silver_players
                        .join(
                            silver_money_events                      
                            .withColumn('rn',
                            F.row_number().over(player_window))
                            .filter(F.col('rn')==1)
                            .select('player_id', 'balance_after_txn'),
                            on='player_id',
                            how='left'
                            )
                            .withColumn(
                            "current_balance",
                            F.coalesce("balance_after_txn", "balance"))
                            .drop('balance', 'balance_after_txn')
)

silver_players_final.write.mode("overwrite").parquet("./data/silver/players")
silver_sessions_final.write.mode("overwrite").parquet("./data/silver/sessions")
silver_transactions_final.write.mode("overwrite").parquet("./data/silver/transactions")
silver_money_events.write.mode("overwrite").parquet("./data/silver/money_events")