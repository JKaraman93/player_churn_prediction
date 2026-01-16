from src.utils.spark_session import get_spark
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, BooleanType

spark = get_spark()
spark.catalog.clearCache()
players_bronze = spark.read.parquet("./data/bronze/players")
sessions_bronze = spark.read.parquet("./data/bronze/sessions")
transactions_bronze = spark.read.parquet("./data/bronze/transactions")
churn_label_bronze = spark.read.parquet("./data/bronze/churn_labels")

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
    .filter(F.col("success_flag") == True)
    .withColumn(
        "signed_amount", 
        F.when(F.col("transaction_type") == "deposit", F.col("amount"))
         .when(F.col("transaction_type") == "withdrawal", -F.col("amount"))
         .otherwise(F.lit(0.0))
    )
)

silver_transactions = (
    silver_transactions
    .join(
        silver_players.select("player_id", "balance",'registration_date'),
        on="player_id",
        how="left"
    )
        .filter(F.col('transaction_ts') >= F.col('registration_date'))
        .drop(F.col('registration_date'))

)

df_all_transaction = ( silver_sessions
                .select('player_id', 
                    F.col('session_id').alias('event_id'),   
                    F.col('session_date').alias('event_ts'), 
                'signed_amount',
                'balance')
                .unionByName(silver_transactions
                .select('player_id', 
                  F.col('transaction_id').alias('event_id'),   
                    F.col('transaction_ts').alias('event_ts'), 
                    'signed_amount',
                    'balance'))

)

'''
balance_window = (
    Window
    .partitionBy("player_id")
    .orderBy("event_ts")
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)

txn_with_tentative = df_all_transaction.withColumn(
    "tentative_balance",
    F.col("balance") + F.sum("signed_amount").over(balance_window)
)

txn_flagged = txn_with_tentative.withColumn(
    "is_valid_txn",
    F.col("tentative_balance") >= 0
)

txn_neutralized = txn_flagged.withColumn(
    "effective_signed_amount",
    F.when(F.col("is_valid_txn"), F.col("signed_amount"))
     .otherwise(F.lit(0.0))
)

silver_all_transactions = txn_neutralized.withColumn(
    "balance_after_txn",
    F.col("balance") + F.sum("effective_signed_amount").over(balance_window)
)'''

# 1. Define the new schema by taking the existing one and adding the new column

# 2. Update the UDF (remove the decorator from the function)
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
df_processed = df_all_transaction.groupBy("player_id").applyInPandas(
    process_transactions, 
    schema=df_all_transaction.schema
)


silver_transactions_final = (silver_transactions
                       .join(df_processed.select(F.col('event_id').alias('transaction_id'),F.col('balance').alias('balance_after_txn')),
                             how='inner',
                             on='transaction_id')
                            .drop('balance')
)



silver_sessions_final = (silver_sessions
                       .join(df_processed.select(F.col('event_id').alias('session_id'), F.col('balance').alias('balance_after_txn')),
                             how='inner',
                             on='session_id')
                           .drop('balance',) 
)

player_window = (Window
    .partitionBy("player_id")
    .orderBy("event_ts")
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)

silver_players_final = (silver_players
                        .join(
                            df_processed                      
                            .withColumn('rn',
                            F.row_number().over(player_window))
                            .filter(F.col('rn')==1)
                            .select('player_id', F.col('balance').alias('updated_balance')),
                            on='player_id',
                            how='left'
                            )
                            .drop('balance')
)


silver_churn_final = (
    churn_label_bronze
    .join(
        silver_players.select("player_id"),
        on="player_id",
        how="inner"
    )
    .withColumn(
        "reference_date",
        F.to_date("reference_date")
    )
)


silver_players_final.limit(1).show()
silver_sessions_final.limit(1).show()
silver_transactions_final.limit(1).show()
silver_churn_final.limit(1).show()

#silver_players_final.write.mode("overwrite").parquet("./data/silver/players")
#silver_sessions_final.write.mode("overwrite").parquet("./data/silver/sessions")
#silver_transactions_final.write.mode("overwrite").parquet("./data/silver/transactions")
#silver_churn_final.write.mode("overwrite").parquet("./data/silver/churn_labels")
