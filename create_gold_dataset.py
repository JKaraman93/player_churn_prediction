from src.utils.spark_session import get_spark
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd
from pyspark.sql.functions import pandas_udf

spark = get_spark()
spark.catalog.clearCache()
players_silver = spark.read.parquet("./data/silver/players")
sessions_silver = spark.read.parquet("./data/silver/sessions")
transactions_silver = spark.read.parquet("./data/silver/transactions")
churn_label_silver = spark.read.parquet("./data/silver/churn_labels")


