from src.utils.spark_session import get_spark

spark = get_spark()
players_bronze = spark.read.parquet("./data/bronze/players")
sessions_bronze = spark.read.parquet("./data/bronze/sessions")
transactions_bronze = spark.read.parquet("./data/bronze/transactions")
churn_label_bronze = spark.read.parquet("./data/bronze/churn_labels")
 
