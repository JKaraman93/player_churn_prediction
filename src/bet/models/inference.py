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

from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.window import Window
from bet.utils.spark_session import get_spark
from bet.utils.config import DataGenConfig
from bet.utils.logging_utils import get_logger
from bet.models.prepare_data_inference import prepare_num_data_inference
from pyspark.ml.functions import vector_to_array
import mlflow 
from mlflow.tracking import MlflowClient
import sys

logger = get_logger(__name__)


def _compare_dataframes(df1: DataFrame, df2: DataFrame) -> bool:
    """
    Verify two DataFrames contain identical records.
    
    Args:
        df1: First DataFrame to compare
        df2: Second DataFrame to compare
        
    Returns:
        True if DataFrames are identical, raises AssertionError otherwise
    """
    result1 = df1.exceptAll(df2).count() == 0
    result2 = df2.exceptAll(df1).count() == 0
    assert result1 and result2, "DataFrames do not match"
    return True


def _add_risk_levels(predictions: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Add risk level categories to predictions and extract flagged players.
    
    Risk levels:
    - High: p_churn >= 0.8
    - Medium: 0.6 <= p_churn < 0.8
    - Low: 0.4 <= p_churn < 0.6
    - None: p_churn < 0.4
    
    Args:
        predictions: DataFrame with p_churn probabilities
        
    Returns:
        Tuple of (enriched predictions with risk levels, flagged players with predictions=1)
    """
    preds = predictions.select('player_idx','reference_date', 'p_churn', 'prediction')
    preds = preds.withColumn('risk_level', 
        F.when(F.col('p_churn') >= 0.8, 'High')
        .when(F.col('p_churn') >= 0.6, 'Medium')
        .when(F.col('p_churn') >= 0.4, 'Low')
        .otherwise(F.lit('None')))
    
    flagged_players = preds.filter(F.col('prediction') == 1).select('player_idx', 'p_churn')
    return preds, flagged_players


def main(test_date: str) -> None:
    """
    Run daily churn risk inference for all active players.
    
    Pipeline:
    1. Load Gold layer behavior features for test_date
    2. Prepare Silver layer features using prepare_num_data_inference()
    3. Load production model from MLflow
    4. Generate predictions for all players
    5. Add risk level categorization
    6. Log results to MLflow
    
    Args:
        test_date: Inference date in YYYY-MM-DD format
        
    Raises:
        Exception: If Gold/Silver data missing or production model unavailable
    """
    logger.info(f"Starting daily inference for {test_date}")
    
    spark = get_spark()
    spark.catalog.clearCache()
    config = DataGenConfig()
    
    # Load Gold and Silver layer data
    try:
        logger.info("Loading Gold layer data...")
        player_behavior = spark.read.parquet("./data/gold/player_behavior")
        player_snapshot = spark.read.parquet("./data/gold/player_snapshot")
        
        logger.info("Preparing inference features...")
        num_data_inference = prepare_num_data_inference(test_date)
        logger.info(f"Prepared features for {num_data_inference.count()} players")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # Validate feature consistency
    try:
        m1 = player_behavior.filter(F.col('reference_date') == test_date).select('player_idx').join(
            num_data_inference, how='inner', on='player_idx')
        _compare_dataframes(m1, num_data_inference)
        logger.info("Feature consistency validation passed")
    except AssertionError as e:
        logger.error(f"Feature validation failed: {e}")
        raise

    # Load production model
    try:
        logger.info("Loading production model from MLflow...")
        mlflow.set_experiment("daily-inference")
        loaded_model = mlflow.spark.load_model("models:/SparkLogisticRegression_train@production")
        
        client = MlflowClient()
        model_version = client.get_model_version_by_alias(
            name="SparkLogisticRegression_train", 
            alias="production")
        train_run_id = model_version.run_id
        run = mlflow.get_run(train_run_id)
        threshold = float(run.data.params["threshold"])
        logger.info(f"Loaded model version {model_version.version} with threshold {threshold}")
    except Exception as e:
        logger.error(f"Failed to load production model: {e}")
        raise

    # Prepare inference data
    logger.info("Assembling final inference dataset...")
    data_inference_ml = (player_behavior.select('player_idx','reference_date')
        .filter(F.col('reference_date') == test_date)
        .join(num_data_inference, how='inner', on='player_idx')
        .join(player_snapshot.select('player_idx', 'country', 'age_bucket'), 
              on="player_idx", how="inner")
    )
    logger.info(f"Inference dataset size: {data_inference_ml.count()} player-dates")

    # Generate predictions and log results
    logger.info("Running inference...")
    with mlflow.start_run(run_name=test_date):
        test_preds = (loaded_model.transform(data_inference_ml)
            .withColumn("p_churn", F.round(vector_to_array("probability")[1], 2)))
        
        results, flagged_players = _add_risk_levels(test_preds)
        
        num_flagged = flagged_players.count()
        num_total = results.count()
        
        # Log parameters and metrics
        mlflow.log_param("run_date", test_date)
        mlflow.log_param("train_run_id", train_run_id)
        mlflow.log_param("model_version", model_version.version)
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("num_players", num_total)
        mlflow.log_param("num_flagged_players", num_flagged)
        mlflow.log_metric("flagged_rate", num_flagged / num_total if num_total > 0 else 0.0)
        
        # Log results table
        mlflow.log_table(flagged_players.toPandas(), 'flagged_players.json')
        
        logger.info(f"Inference completed: {num_flagged}/{num_total} players flagged for churn")
        logger.info(f"Results logged to MLflow run")
        
        # Display summary
        results.show(3)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <test_date>")
        print("  Example: python inference.py 2024-03-15")
        sys.exit(1)
    
    test_date = sys.argv[1]
    logger.info(f"Initiating inference pipeline for {test_date}")
    main(test_date)

