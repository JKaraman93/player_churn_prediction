"""
Backtest: Model Evaluation on Test Data

This module evaluates the trained logistic regression model on held-out test data.
It loads the production model from MLflow, runs inference on test data, and generates
evaluation metrics (precision, recall, F1, AUPR) for performance assessment.

The backtest uses the same feature engineering pipeline as training to ensure
consistency and avoid data leakage.

Outputs:
- MLflow run with test metrics and predictions
- Test data predictions with churn probabilities and segments
- Calibration analysis comparing predicted vs actual churn rates
"""

from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.window import Window
from bet.utils.spark_session import get_spark
from bet.utils.config import DataGenConfig
from bet.utils.logging_utils import get_logger
from pyspark.ml.functions import vector_to_array
import mlflow 
from mlflow.tracking import MlflowClient

logger = get_logger(__name__)


def main() -> None:
    """
    Run model backtest on held-out test data.
    
    Pipeline:
    1. Load production model from MLflow with metadata
    2. Extract training parameters (threshold, run_id, date range)
    3. Load Gold layer test data for specified date range
    4. Generate predictions on test set
    5. Compute evaluation metrics (precision, recall, F1, AUPR)
    6. Analyze calibration (predicted probability vs actual churn rate)
    7. Log results to MLflow
    
    Metrics tracked:
    - Daily precision, recall, F1
    - Churn rate by risk level (high/medium/low/none)
    - Calibration statistics
    - Confusion matrices
    
    Returns:
        None - results logged to MLflow
        
    Raises:
        Exception: If production model or test data unavailable
    """
    logger.info("Starting model backtest")
    
    spark = get_spark()
    spark.catalog.clearCache()
    
    # Load production model and parameters
    try:
        logger.info("Loading production model from MLflow...")
        mlflow.set_experiment("backtesting")
        loaded_model = mlflow.spark.load_model("models:/SparkLogisticRegression_train@production")
        
        client = MlflowClient()
        model_version = client.get_model_version_by_alias(
            name="SparkLogisticRegression_train",
            alias="production")
        train_run_id = model_version.run_id
        run = mlflow.get_run(train_run_id)
        
        start_date = run.data.params["test_start"]
        end_date = run.data.params["test_end"]
        threshold = float(run.data.params["threshold"])
        logger.info(f"Model: version {model_version.version}, threshold {threshold}")
        logger.info(f"Test period: {start_date} to {end_date}")
    except Exception as e:
        logger.error(f"Failed to load production model: {e}")
        raise

    # Load test data
    try:
        logger.info("Loading test data from Gold layer...")
        gold_tables = read_gold_tables(spark)
        player_behavior = gold_tables['player_behavior']
        player_snapshot = gold_tables['player_snapshot']
        labels = gold_tables['labels']
        
        test_df = (player_behavior.filter(F.col('reference_date') >= start_date)
                .filter(F.col('reference_date') <= end_date)
                .join(player_snapshot.select('player_idx', 'country', 'age_bucket'), 
                      on="player_idx", how="left")
                .join(labels, on=["player_idx", "reference_date"], how="inner")
                .withColumn("next_7d_churn_idx", F.col("next_7d_churn").cast("int")))
        
        test_count = test_df.count()
        logger.info(f"Loaded {test_count} test records spanning {start_date} to {end_date}")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        raise

    # Run backtest inference
    logger.info("Generating predictions on test set...")
    back_test_preds = (loaded_model.transform(test_df)
        .withColumn("p_churn", F.round(vector_to_array("probability")[1], 2))
        .withColumn('risk_level', 
            F.when(F.col('p_churn') >= 0.8, 'High')
            .when(F.col('p_churn') >= 0.6, 'Medium')
            .when(F.col('p_churn') >= 0.4, 'Low')
            .otherwise(F.lit('None')))
    )
    
    logger.info(f"Generated predictions for {back_test_preds.count()} test samples")

    # Compute daily metrics
    logger.info("Computing daily performance metrics...")
    pred_per_day = (back_test_preds.groupBy('reference_date')
        .agg(
            F.count('player_idx').alias('num_players'),
            F.sum(F.when(F.col("prediction") == 1, 1).otherwise(F.lit(0))).alias("num_flagged"),
            F.sum(F.when(F.col("next_7d_churn_idx") == 1, 1).otherwise(F.lit(0))).alias("num_churned"),
            F.sum(F.when(((F.col("next_7d_churn_idx") == 1) & (F.col("prediction") == 1)), 1).otherwise(F.lit(0))).alias("tp"),
            F.sum(F.when(((F.col("next_7d_churn_idx") == 1) & (F.col("prediction") == 0)), 1).otherwise(F.lit(0))).alias("fn"),
            F.sum(F.when(((F.col("next_7d_churn_idx") == 0) & (F.col("prediction") == 0)), 1).otherwise(F.lit(0))).alias("tn"),
            F.sum(F.when(((F.col("next_7d_churn_idx") == 0) & (F.col("prediction") == 1)), 1).otherwise(F.lit(0))).alias("fp"),
            # Churn by risk level
            F.sum(F.when(((F.col("next_7d_churn_idx") == 1) & (F.col("risk_level") == 'High')), 1).otherwise(F.lit(0))).alias("num_churned_high_risk"),
            F.sum(F.when(((F.col("next_7d_churn_idx") == 1) & (F.col("risk_level") == 'Medium')), 1).otherwise(F.lit(0))).alias("num_churned_med_risk"),
            F.sum(F.when(((F.col("next_7d_churn_idx") == 1) & (F.col("risk_level") == 'Low')), 1).otherwise(F.lit(0))).alias("num_churned_low_risk"),
            F.sum(F.when(((F.col("next_7d_churn_idx") == 1) & (F.col("risk_level") == 'None')), 1).otherwise(F.lit(0))).alias("num_churned_no_risk"),
        )
        .withColumn('num_churned', F.col('tp') + F.col('fn'))
        .withColumn('precision', F.when(F.col('tp') + F.col('fp') > 0, 
                        F.round(F.col('tp') / (F.col('tp') + F.col('fp')), 2)).otherwise(F.lit(0)))
        .withColumn('recall', F.when(F.col('tp') + F.col('fn') > 0, 
                        F.round(F.col('tp') / (F.col('tp') + F.col('fn')), 2)).otherwise(F.lit(0)))
        .withColumn('f1', F.when(F.col('precision') + F.col('recall') > 0, 
                        F.round(2 * F.col('precision') * F.col('recall') / (F.col('precision') + F.col('recall')), 2))
                        .otherwise(F.lit(0)))
        .withColumn('churned_rate_high_risk', F.round(F.col('num_churned_high_risk') / F.col('num_churned'), 2))
        .withColumn('churned_rate_med_risk', F.round(F.col('num_churned_med_risk') / F.col('num_churned'), 2))
        .withColumn('churned_rate_low_risk', F.round(F.col('num_churned_low_risk') / F.col('num_churned'), 2))
        .withColumn('churned_rate_no_risk', F.round(F.col('num_churned_no_risk') / F.col('num_churned'), 2))
        .drop('tp', 'fn', 'tn', 'fp')
    ).orderBy("reference_date")
    
    select_cols = ['precision', 'recall', 'f1', 'churned_rate_high_risk', 'churned_rate_med_risk', 'churned_rate_low_risk', 'churned_rate_no_risk']    
    df_avg = pred_per_day.select([F.round(F.avg(c), 2).alias('avg_' + c) for c in select_cols])
    avg_metrics = df_avg.first().asDict()
    
    logger.info(f"Average metrics - Precision: {avg_metrics.get('avg_precision')}, Recall: {avg_metrics.get('avg_recall')}, F1: {avg_metrics.get('avg_f1')}")

    # Compute calibration curve
    logger.info("Computing model calibration...")
    preds = back_test_preds.withColumn("prob_bin", F.floor(F.col("p_churn") * 10) / 10)
    calibration = (preds.groupBy("prob_bin")
        .agg(F.round(F.avg("next_7d_churn_idx"), 2).alias("actual_rate"), 
             F.count("*").alias("players"))
        .orderBy("prob_bin")
    )

    # Log results to MLflow
    logger.info("Logging results to MLflow...")
    with mlflow.start_run(run_name='backtest_final'):
        mlflow.log_param('start', start_date)
        mlflow.log_param('end', end_date)
        mlflow.log_param("train_run_id", train_run_id)
        mlflow.log_param("model_version", model_version.version)
        mlflow.log_param("threshold", threshold)
        
        # Log metrics
        mlflow.log_metrics(avg_metrics)
        
        # Log tables
        mlflow.log_table(pred_per_day.toPandas(), 'daily_metrics.json')
        mlflow.log_table(calibration.toPandas(), 'calibration.json')
        
        logger.info("Backtest completed and results logged to MLflow")


if __name__ == "__main__":
    logger.info("Starting backtest evaluation")
    main()


