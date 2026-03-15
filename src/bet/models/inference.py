"""
Daily Inference Pipeline: Production Churn Risk Scoring

This module runs daily batch inference to generate churn risk scores for all players.
It loads the production model from MLflow, computes rolling features from Silver data,
and generates daily predictions with churn probabilities and risk segments.

Designed for daily execution with a test_date parameter to ensure production-ready
feature consistency and prevent data leakage.

Usage:
    python inference.py <test_date>  # e.g., python inference.py 2024-03-15

Outputs:
- Daily churn risk predictions with scores and segments
- Player-level risk classifications (low, medium, high)
"""

from bet.utils.spark_session import get_spark
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd
from pyspark.sql.functions import pandas_udf
import bet.utils.config as config
from bet.ingestion.last_activity_generator import generate_last_activity
from bet.models.prepare_data_inference import prepare_num_data_inference
from pyspark.ml.functions import vector_to_array
import mlflow 
import numpy as np
from mlflow.tracking import MlflowClient
import sys

if len(sys.argv) < 2:
    print("Usage: python my_script.py <your_argument>")
    sys.exit(1)

test_date = sys.argv[1]
print(f"Predictions for : {test_date}")

def compare(df1,df2): 
   assert (( df1.exceptAll(df2).count() == 0) & (df2.exceptAll(df1).count() == 0))

def result_display(preds):
    preds = preds.select('player_idx','reference_date', 'p_churn', 'prediction')
    preds = preds.withColumn('risk_level', 
        F.when(F.col('p_churn')>=0.8, 'High')
        .when(F.col('p_churn')>=0.6, 'Medium')
        .when(F.col('p_churn')>=0.4, 'Low')
        .otherwise(F.lit('None')))
    flagged_players = preds.filter(F.col('prediction')== 1).select('player_idx', 'p_churn')
    return preds, flagged_players

spark = get_spark()
spark.catalog.clearCache()
config_ = config.DataGenConfig()

player_behavior = spark.read.parquet("./data/gold/player_behavior")
player_snapshot = spark.read.parquet("./data/gold/player_snapshot")
labels = spark.read.parquet("./data/gold/labels")

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

num_data_inference = prepare_num_data_inference(test_date) ## Only transactions features should be printed
#num_data_inference.show(3)

m1 = player_behavior.filter(F.col('reference_date')==test_date).select('player_idx').join(num_data_inference, how='inner', on='player_idx', )
compare(m1, num_data_inference)

mlflow.set_experiment("daily-inference")
#mlflow.set_tracking_uri("file:./mlruns")
loaded_model = mlflow.spark.load_model( "models:/SparkLogisticRegression_train@production")

client = MlflowClient()
model_version = client.get_model_version_by_alias( name="SparkLogisticRegression_train", alias="production")
run_id = model_version.run_id
run = mlflow.get_run(run_id)
threshold = run.data.params["threshold"]
print(run_id)

data_inference_ml = (player_behavior.select('player_idx','reference_date').filter(F.col('reference_date')==test_date)
.join(num_data_inference, how ='inner', on='player_idx')
.join(player_snapshot.select('player_idx', 'country', 'age_bucket'), on="player_idx", how="inner")
)
with mlflow.start_run(run_name=test_date):
    test_preds = (loaded_model.transform(data_inference_ml)
    .withColumn("p_churn", F.round(vector_to_array("probability")[1],2)))
    results, flagged_players = result_display(test_preds)
    mlflow.log_param("run_date", test_date)                # the inference date
    mlflow.log_param("run_id", train_run_id)                
    mlflow.log_param("model", model_version.source)               
    mlflow.log_param("threshold", threshold)          # threshold used for labeling
    mlflow.log_param("num_players", data_inference_ml.count())
    mlflow.log_param("num_flagged_players", flagged_players.count())
    mlflow.log_table(flagged_players.toPandas(), 'flagged_players.json')
    results.show()

