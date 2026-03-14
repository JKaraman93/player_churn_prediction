from src.utils.spark_session import get_spark
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd
from pyspark.sql.functions import pandas_udf
import src.utils.config as config
from src.ingestion.last_activity_generator import generate_last_activity
from prepare_data_inference import prepare_num_data_inference
from pyspark.ml.functions import vector_to_array
import mlflow 
import numpy as np
from mlflow.tracking import MlflowClient

spark = get_spark()
spark.catalog.clearCache()
config_ = config.DataGenConfig()

player_behavior = spark.read.parquet("./data/gold/player_behavior")
player_snapshot = spark.read.parquet("./data/gold/player_snapshot")
labels = spark.read.parquet("./data/gold/labels")
mlflow.set_experiment("backtesting")

#mlflow.set_tracking_uri("file:./mlruns")
loaded_model = mlflow.spark.load_model( "models:/SparkLogisticRegression_train@production")
client = MlflowClient()
model_version = client.get_model_version_by_alias( name="SparkLogisticRegression_train",  alias="production")
run_id = model_version.run_id
run = mlflow.get_run(run_id)
print(run_id)

######### Backtest using test data ###########

start = run.data.params["test_start"]
end = run.data.params["test_end"]
threshold = run.data.params["threshold"]

test_df =  (player_behavior.filter(F.col('reference_date')>=start)
        .join(player_snapshot.select('player_idx', 'country', 'age_bucket'), on="player_idx", how="left")
        .join(labels, on=["player_idx", "reference_date"], how="inner")
         .withColumn("next_7d_churn_idx", F.col("next_7d_churn").cast("int")))

print(start, end)

with mlflow.start_run(run_name='test_data'):
    mlflow.log_param('start',start)
    mlflow.log_param('end',end)
    mlflow.log_param("train_run_id", run_id)                # the inference date
    mlflow.log_param("model", model_version.source)       
    mlflow.log_param("threshold", threshold)          # threshold used for labeling

    back_test_preds = (loaded_model.transform(test_df)
            .withColumn("p_churn", vector_to_array("probability")[1])
            .withColumn('risk_level', 
                    F.when(F.col('p_churn')>=0.8, 'High')
                    .when(F.col('p_churn')>=0.6, 'Medium')
                    .when(F.col('p_churn')>=0.4, 'Low')
                    .otherwise(F.lit('None')))
                    )
    #back_test_preds.show(4)

    pred_per_day = (back_test_preds.groupBy('reference_date')
        .agg(
            F.count('player_idx').alias('num_players'),
            F.sum(F.when(F.col("prediction") == 1, 1).otherwise(F.lit(0))).alias("num_flagged"),
            F.sum(F.when(F.col("next_7d_churn_idx") == 1, 1).otherwise(F.lit(0))).alias("num_churned"),
            F.sum(F.when(((F.col("next_7d_churn_idx") == 1) & (F.col("prediction") == 1)), 1).otherwise(F.lit(0))).alias("tp"),
            F.sum(F.when(((F.col("next_7d_churn_idx") == 1) & (F.col("prediction") == 0)), 1).otherwise(F.lit(0))).alias("fn"),
            F.sum(F.when(((F.col("next_7d_churn_idx") == 0) & (F.col("prediction") == 0)), 1).otherwise(F.lit(0))).alias("tn"),
            F.sum(F.when(((F.col("next_7d_churn_idx") == 0) & (F.col("prediction") == 1)), 1).otherwise(F.lit(0))).alias("fp"), 
            F.sum(F.when(((F.col("next_7d_churn_idx") == 1) & (F.col("risk_level") == 'High')), 1).otherwise(F.lit(0))).alias("num_churned_high_risk"), 
            F.sum(F.when(((F.col("next_7d_churn_idx") == 1) & (F.col("risk_level") == 'Medium')), 1).otherwise(F.lit(0))).alias("num_churned_med_risk"), 
            F.sum(F.when(((F.col("next_7d_churn_idx") == 1) & (F.col("risk_level") == 'Low')), 1).otherwise(F.lit(0))).alias("num_churned_low_risk"), 
            F.sum(F.when(((F.col("next_7d_churn_idx") == 1) & (F.col("risk_level") == 'None')), 1).otherwise(F.lit(0))).alias("num_churned_no_risk"), 
        )
        .withColumn('num_churned', F.col('tp') + F.col('fn'))
        .withColumn('precision', F.when(F.col('tp') + F.col('fp') > 0, F.round(F.col('tp') / (F.col('tp')+ F.col('fp')) ,2)).otherwise(F.lit(0)))
        .withColumn('recall', F.when(F.col('tp') + F.col('fn') > 0, F.round(F.col('tp') / (F.col('tp')+ F.col('fn')) ,2)).otherwise(F.lit(0)))
        .withColumn('f1', F.when(F.col('precision') + F.col('recall') > 0, F.round(2 * F.col('precision') * F.col('recall') / (F.col('precision')+ F.col('recall')) ,2)).otherwise(F.lit(0)))
        .withColumn('churned_rate_high_risk',F.round(F.col('num_churned_high_risk') / F.col('num_churned'),2))
        .withColumn('churned_rate_med_risk', F.round(F.col('num_churned_med_risk') / F.col('num_churned'),2))
        .withColumn('churned_rate_low_risk', F.round(F.col('num_churned_low_risk') / F.col('num_churned'),2))
        .withColumn('churned_rate_no_risk', F.round(F.col('num_churned_no_risk') / F.col('num_churned'),2))
    .drop('tp', 'fn', 'tn','fp')

    ).orderBy("reference_date")
    select_cols = ['precision','recall', 'f1','churned_rate_high_risk','churned_rate_med_risk','churned_rate_low_risk', 'churned_rate_no_risk']    
    
    df_avg = pred_per_day.select([F.round(F.avg(c),2).alias('avg_'+ c) for c in select_cols])
    mlflow.log_metrics(df_avg.first().asDict())

    preds = back_test_preds.withColumn("prob_bin",    F.floor(F.col("p_churn") * 10) / 10) 
    calibration = (preds.groupBy("prob_bin")
        .agg(F.round(F.avg("next_7d_churn_idx"),2).alias("actual_rate"), F.count("*").alias("players") )
        .orderBy("prob_bin")  ) 
    mlflow.log_table(calibration.toPandas(),'calibration.json')


