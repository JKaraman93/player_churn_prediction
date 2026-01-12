from src.utils.spark_session import get_spark
from pyspark.sql import functions as F
from pyspark.sql.window import Window


spark = get_spark()
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
)

silver_churn = (
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


silver_players.write.mode("overwrite").parquet("./data/silver/players")
silver_sessions.write.mode("overwrite").parquet("./data/silver/sessions")
silver_transactions.write.mode("overwrite").parquet("./data/silver/transactions")
silver_churn.write.mode("overwrite").parquet("./data/silver/churn_labels")
