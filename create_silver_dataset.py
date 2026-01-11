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
        silver_players.select("player_id"),
        on="player_id",
        how="inner"
    )
    .filter(F.col("session_duration_sec") >= 0)
    .withColumn("bet_count", F.coalesce(F.col("bet_count"), F.lit(0)))
    .withColumn("total_bet_amount", F.coalesce(F.col("total_bet_amount"), F.lit(0.0)))
    .withColumn("total_win_amount", F.coalesce(F.col("total_win_amount"), F.lit(0.0)))
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
        silver_players.select("player_id", "balance"),
        on="player_id",
        how="left"
    )
)


balance_window = (
    Window
    .partitionBy("player_id")
    .orderBy("transaction_ts")
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)

silver_transactions = (
    silver_transactions
    .withColumn(
        "balance_after_txn",
        F.col("balance") + F.sum("signed_amount").over(balance_window)
    )
)

silver_transactions = silver_transactions.filter(
    F.col("balance_after_txn") >= 0
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
