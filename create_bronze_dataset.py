from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
import random
import src.utils.config as config
import os
from src.ingestion.generate_players import generate_player_profiles
from src.ingestion.player_lifecycle import assign_lifecycle
from src.ingestion.generate_sessions import generate_gameplay_sessions
from src.ingestion.generate_transactions import generate_financial_transactions
from src.ingestion.churn_label_generator import generate_churn_labels
from ingestion.generate_initial_balance import assign_balance

config_ = config.DataGenConfig()

# Set logging level to reduce console warnings
spark = SparkSession.builder.master("local[*]").appName('app_name').getOrCreate()
spark.sparkContext.setLogLevel("ERROR") 

df_players = generate_player_profiles(spark, config_)
df_players = assign_lifecycle(df_players)
df_players = assign_balance(df_players,config_)

df_sessions = generate_gameplay_sessions(df_players, spark, config_)
df_money_transactions = generate_financial_transactions(df_players)
df_churn = generate_churn_labels(df_sessions, config_)

df_players.write.mode("append").parquet("./data/bronze/players")
df_sessions.write.mode("append").partitionBy("session_date").parquet("./data/bronze/sessions")
df_money_transactions.write.mode("append").parquet("./data/bronze/transactions")
df_churn.write.mode("append").parquet("./data/bronze/churn_labels")

print ('end')