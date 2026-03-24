"""
Model Training: Logistic Regression Pipeline

This module implements the main churn prediction model training workflow.
It builds a Spark ML pipeline with feature preprocessing (encoding, scaling),
trains a logistic regression classifier, performs cross-validation, and optimizes
the decision threshold for business-aligned classification.

Key features:
- Handles class imbalance with class weights
- Computes precision-recall curves and selects optimal threshold
- Logs hyperparameters, metrics, and model artifacts to MLflow
- Ensures full reproducibility and experiment tracking

Metrics tracked:
- Area Under Precision-Recall Curve (AUPR) - primary metric
- Precision, Recall, F1 Score
- Confusion matrix components

Outputs:
- Trained Spark ML pipeline model
- MLflow experiment run with all metrics and parameters
- Threshold configuration for risk segmentation
"""

import sys
import platform
import subprocess
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from mlflow.models.signature import infer_signature
import mlflow
from datetime import datetime

from bet.utils.spark_session import get_spark
from bet.utils.config import DataGenConfig
from bet.utils.logging_utils import get_logger
from bet.utils.data_utils import read_gold_tables

logger = get_logger(__name__)


def compute_metrics(df: DataFrame, threshold: float) -> Tuple[float, float, float, float, float]:
    """
    Compute binary classification metrics at a given threshold.
    
    Args:
        df: DataFrame with 'p_churn' (probability) and 'next_7d_churn_idx' (label) columns
        threshold: Classification threshold (0 to 1)
        
    Returns:
        Tuple of (precision, recall, f1, day_avg_churned, day_avg_flagged)
        - precision: TP / (TP + FP)
        - recall: TP / (TP + FN)
        - f1: Harmonic mean of precision and recall
        - day_avg_churned: Average daily churn count
        - day_avg_flagged: Average daily positive predictions
        
    Raises:
        Exception: If threshold not in valid range [0, 1]
    """
    if not (0 <= threshold <= 1):
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    
    pred = df.withColumn(
        "pred_label",
        (F.col("p_churn") >= threshold).cast("int")
    )

    pred_per_day = (pred.groupBy('reference_date')
    .agg(
        #F.sum('player_idx').alias('num_players'),
        F.sum(F.when(F.col("pred_label") == 1, 1).otherwise(F.lit(0))).alias("num_flagged"),
        F.sum(F.when(F.col("next_7d_churn_idx") == 1, 1).otherwise(F.lit(0))).alias("num_churned")
        )
    )

    select_cols = ['num_churned', 'num_flagged']
    day_average = pred_per_day.select([F.round(F.avg(c), 0).alias(c) for c in select_cols])
    day_avg_churned, day_avg_flagged = day_average.first()

    metrics = pred.groupBy("next_7d_churn_idx", "pred_label").count()
    tp = metrics.filter("next_7d_churn_idx = 1 AND pred_label = 1").select("count").first()
    fp = metrics.filter("next_7d_churn_idx = 0 AND pred_label = 1").select("count").first()
    fn = metrics.filter("next_7d_churn_idx = 1 AND pred_label = 0").select("count").first()

    tp = tp[0] if tp else 0
    fp = fp[0] if fp else 0
    fn = fn[0] if fn else 0

    precision = float(np.round(tp / (tp + fp), 2)) if (tp + fp) > 0 else 0.0
    recall = float(np.round(tp / (tp + fn), 2)) if (tp + fn) > 0 else 0.0
    f1 = (float(np.round(2 * precision * recall / (precision + recall), 2))
        if (precision + recall) > 0 else 0.0
    )

    return precision, recall, f1, day_avg_churned, day_avg_flagged


def add_class_weight(df: DataFrame, weight_for_churn: float) -> DataFrame:
    """
    Add class weight column to handle class imbalance.
    
    Minority class (churn) gets higher weight to balance training.
    
    Args:
        df: Input DataFrame with 'next_7d_churn' column
        weight_for_churn: Weight multiplier for positive class
        
    Returns:
        DataFrame with added 'class_weight' column
    """
    return df.withColumn(
        "class_weight",
        F.when(F.col("next_7d_churn"), weight_for_churn).otherwise(1.0)
    )


def _prepare_data(spark: SparkSession, sample_fraction: float = 1.0) -> Tuple[DataFrame, DataFrame, DataFrame, List[str], List[str]]:
    """
    Load data from Gold layer and split into train/val/test sets.
    
    Splits chronologically:
    - Train: 0-70% of dates
    - Val: 70-85% of dates
    - Test: 85-100% of dates
    
    Args:
        spark: Active Spark session
        sample_fraction: Fraction of players to sample (default 1.0 = all data)
        
    Returns:
        Tuple of (train_df, val_df, test_df, numeric_cols, categorical_cols)
        
    Raises:
        Exception: If data files cannot be loaded
    """
    logger.info("Loading data from Gold layer...")
    try:
        gold_tables = read_gold_tables(spark)
        player_behavior = gold_tables['player_behavior']
        player_snapshot = gold_tables['player_snapshot']
        labels = gold_tables['labels']
        logger.info(f"Loaded Gold tables: {player_behavior.count()} behavior records")
    except Exception as e:
        logger.error(f"Failed to load data from Gold layer: {e}")
        raise

    # Feature columns
    numeric_cols = [
        "balance_7d_ago", "balance_30d_ago", "net_amount_result_7d",
        "net_amount_result_30d", "num_sessions_7d", "num_sessions_30d",
        "avg_sessions_duration_30d", "avg_bet_amount_30d",
        "net_game_result_7d", "net_game_result_30d",
        "failed_withdrawals_30d", "deposit_count_30d", "withdrawal_count_30d",
        "withdrawal_ratio"
    ]
    categorical_cols = ["country", "age_bucket"]

    # Data preparation
    player_snapshot_selected = player_snapshot.select('player_idx', 'country', 'age_bucket')
    sample_players = player_snapshot_selected.select("player_idx").sample(sample_fraction)
    
    model_df = (
        player_behavior
        .join(player_snapshot_selected, on="player_idx", how="left")
        .join(labels, on=["player_idx", "reference_date"], how="inner")
    )

    sample_dataset = sample_players.join(model_df, on="player_idx", how="inner") \
                                   .withColumn("next_7d_churn_idx", F.col("next_7d_churn").cast("int"))

    # Chronological train/val/test split
    logger.info("Performing chronological train/val/test split...")
    dates = [row.reference_date for row in sample_dataset.select("reference_date").distinct().orderBy("reference_date").collect()]
    n = len(dates)
    train_cut = dates[int(n * 0.70)]
    val_cut = dates[int(n * 0.85)]

    train_df = sample_dataset.filter(F.col("reference_date") < train_cut)
    val_df = sample_dataset.filter((F.col("reference_date") >= train_cut) & (F.col("reference_date") < val_cut))
    test_df = sample_dataset.filter(F.col("reference_date") >= val_cut)

    logger.info(f"Train: {train_df.count()} rows ({dates[0]} to {train_cut})")
    logger.info(f"Val: {val_df.count()} rows ({train_cut} to {val_cut})")
    logger.info(f"Test: {test_df.count()} rows ({val_cut} to {dates[-1]})")

    return train_df, val_df, test_df, numeric_cols, categorical_cols


def _build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[Pipeline, BinaryClassificationEvaluator]:
    """
    Construct feature engineering and ML pipeline.
    
    Pipeline stages:
    1. StringIndexer: Encode categorical features
    2. OneHotEncoder: Expand categories
    3. VectorAssembler: Combine numeric features
    4. StandardScaler: Normalize numeric features
    5. VectorAssembler: Final feature vector
    6. LogisticRegression: Binary classifier with class weighting
    
    Args:
        numeric_cols: List of numeric feature column names
        categorical_cols: List of categorical feature column names
        
    Returns:
        Tuple of (pipeline, evaluator)
        - pipeline: Spark ML Pipeline for training
        - evaluator: BinaryClassificationEvaluator (AUPR metric)
    """
    logger.info("Building Spark ML pipeline...")
    
    categorical_idx = [c + "_idx" for c in categorical_cols]
    categorical_ohe = [c + "_ohe" for c in categorical_cols]

    # Feature engineering stages
    indexer = StringIndexer(inputCols=categorical_cols, outputCols=categorical_idx, handleInvalid="error")
    ohe = OneHotEncoder(inputCols=categorical_idx, outputCols=categorical_ohe, dropLast=False)
    numeric_assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
    scaler = StandardScaler(inputCol="numeric_features", outputCol="numeric_features_scaled", withMean=True, withStd=True)
    final_assembler = VectorAssembler(inputCols=["numeric_features_scaled"] + categorical_ohe, outputCol="features")

    # Logistic regression with class weights
    lr = LogisticRegression(
        featuresCol="features", 
        labelCol="next_7d_churn_idx", 
        weightCol="class_weight", 
        maxIter=50
    )

    pipeline = Pipeline(stages=[indexer, ohe, numeric_assembler, scaler, final_assembler, lr])
    evaluator = BinaryClassificationEvaluator(labelCol="next_7d_churn_idx", metricName="areaUnderPR")

    logger.info(f"Pipeline ready: {len(numeric_cols)} numeric + {len(categorical_cols)} categorical features")
    return pipeline, evaluator


def _tune_hyperparams(pipeline: Pipeline, train_df: DataFrame, evaluator: BinaryClassificationEvaluator, parent_run_id: str = None) -> Tuple[PipelineModel, float, float, str]:
    """
    Perform grid search cross-validation on training set.
    
    Tests combinations of:
    - regParam: [0.01, 0.1, 0.5]
    - elasticNetParam: [0.0, 0.5]
    
    Uses 3-fold CV with AUPR as evaluation metric.
    
    Args:
        pipeline: Unfitted Spark ML pipeline
        train_df: Training data with class weights
        evaluator: AUPR evaluator
        parent_run_id: Optional parent run ID for nested run tracking
        
    Returns:
        Tuple of (best_model, best_reg_param, best_elastic_param, run_id)
        
    Raises:
        Exception: If CV training fails
    """
    logger.info("Starting hyperparameter tuning with 3-fold cross-validation...")
    
    paramGrid = ParamGridBuilder() \
        .addGrid(pipeline.getStages()[-1].regParam, [0.01, 0.1, 0.5]) \
        .addGrid(pipeline.getStages()[-1].elasticNetParam, [0.0, 0.5]) \
        .build()

    cv = CrossValidator(
        estimator=pipeline, 
        estimatorParamMaps=paramGrid, 
        evaluator=evaluator, 
        numFolds=3, 
        parallelism=2
    )

    try:
        with mlflow.start_run(run_name='hyperparameter_tuning', nested=parent_run_id is not None) as cv_run:
            # Link to parent run if provided
            if parent_run_id:
                mlflow.set_tag("parent_run_id", parent_run_id)
                mlflow.set_tag("phase", "hyperparameter_tuning")
            
            mlflow.log_param('cv_folds', 3)
            mlflow.log_param('cv_parallelism', 2)
            
            cv_model = cv.fit(train_df)
            best_model = cv_model.bestModel
            lr_best = best_model.stages[-1]
            best_reg = lr_best.getRegParam()
            best_elastic = lr_best.getElasticNetParam()

            mlflow.log_param("regParam", best_reg)
            mlflow.log_param("elasticNetParam", best_elastic)

            # Log training metrics
            train_preds = best_model.transform(train_df).withColumn("p_churn", vector_to_array("probability")[1])
            train_aupr = evaluator.evaluate(train_preds)
            mlflow.log_metric("train_aupr", train_aupr)
            
            logger.info(f"Best params: regParam={best_reg}, elasticNetParam={best_elastic}, AUPR={train_aupr:.3f}")

            # Log feature importance
            _log_feature_importance(best_model, train_df)

            # Register initial model
            sample_input = train_df.limit(100).toPandas()
            sample_output = train_preds.select("p_churn").limit(100).toPandas()
            signature = infer_signature(sample_input, sample_output)

            mlflow.spark.log_model(
                spark_model=best_model,
                artifact_path='spark_model',
                registered_model_name='SparkLogisticRegression_initial_train',
                signature=signature,
            )

            run_id = cv_run.info.run_id
            logger.info(f"Model registered with run_id: {run_id}")

        return best_model, best_reg, best_elastic, run_id

    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        raise


def _optimize_threshold(best_model: PipelineModel, val_df: DataFrame, numeric_cols: List[str], categorical_cols: List[str], parent_run_id: str = None) -> Tuple[float, float, float, float, float]:
    """
    Find optimal decision threshold using validation set.
    
    Tests 19 thresholds (5% to 95%) and selects based on F1 score.
    Returns alternates based on other criteria:
    - th_rec_f1_05: Max recall with F1 > 0.5
    - th_rec_08: Min threshold maintaining recall > 0.8
    - th_flagged_players: Predicted flagged close to actual churned
    
    Args:
        best_model: Fitted pipeline model from CV training
        val_df: Validation data with class weights
        numeric_cols: Numeric feature names
        categorical_cols: Categorical feature names
        parent_run_id: Optional parent run ID for nested run tracking
        
    Returns:
        Tuple of (th_f1, th_rec_f1_05, th_rec_08, th_flagged_players, best_model_uri)
        
    Raises:
        Exception: If threshold optimization fails
    """
    logger.info("Optimizing decision threshold on validation set...")
    
    thresholds = [i / 100 for i in range(5, 96, 5)]
    
    try:
        with mlflow.start_run(run_name='threshold_optimization', nested=parent_run_id is not None) as opt_run:
            # Link to parent run if provided
            if parent_run_id:
                mlflow.set_tag("parent_run_id", parent_run_id)
                mlflow.set_tag("phase", "threshold_optimization")
            
            results = []
            val_preds = best_model.transform(val_df).withColumn("p_churn", vector_to_array("probability")[1])
            val_preds.persist()
            val_preds.count()
            
            for t in thresholds:
                precision, recall, f1, day_avg_churned, day_avg_flagged = compute_metrics(val_preds, t)
                results.append({
                    'threshold': t, 
                    'precision': precision, 
                    'recall': recall, 
                    'f1': f1,  
                    'day_avg_churned': day_avg_churned, 
                    'day_avg_flagged': day_avg_flagged
                })
                logger.info(f"Threshold {t:.2f}: precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f}")

            mlflow.log_table(pd.DataFrame(results), "threshold_metrics.json")
            
            # Select thresholds based on different criteria
            metrics_df = get_spark().createDataFrame(results)
            th_f1 = metrics_df.orderBy(F.desc("f1")).first()["threshold"]
            th_rec_f1_05 = metrics_df.filter(F.col('f1') > 0.5).orderBy(F.desc("recall")).first()["threshold"]
            th_rec_08 = metrics_df.filter(F.col('recall') > 0.8).orderBy(F.asc("recall")).first()["threshold"]
            th_flagged_players = (metrics_df.filter(F.col('day_avg_flagged') > F.col('day_avg_churned'))
                .orderBy(F.asc('day_avg_flagged'))).first()["threshold"]

            logger.info(f"Selected thresholds: F1={th_f1:.2f}, Rec(F1>0.5)={th_rec_f1_05:.2f}, Rec>0.8={th_rec_08:.2f}, Flagged={th_flagged_players:.2f}")

            # Plot PR curve
            pdf = val_preds.select("p_churn", "next_7d_churn_idx").toPandas()
            precision_arr, recall_arr, _ = precision_recall_curve(
                pdf["next_7d_churn_idx"],
                pdf["p_churn"]
            )
            plt.figure()
            plt.plot(recall_arr, precision_arr)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve (Validation)")
            mlflow.log_figure(plt.gcf(), "pr_curve.png")
            plt.close()

            val_preds.unpersist()

        return th_f1, th_rec_f1_05, th_rec_08, th_flagged_players, opt_run.info.run_id

    except Exception as e:
        logger.error(f"Threshold optimization failed: {e}")
        raise


def _log_feature_importance(model: PipelineModel, train_df: DataFrame) -> None:
    """
    Extract and log feature coefficients from trained logistic regression.
    
    Args:
        model: Fitted pipeline model
        train_df: Training data (used for OHE category sizes)
    """
    lr_model = model.stages[-1]
    ohe_model = model.stages[1]
    
    coeffs = lr_model.coefficients.toArray().tolist()
    
    # Reconstruct feature names
    numeric_cols = [
        "balance_7d_ago", "balance_30d_ago", "net_amount_result_7d",
        "net_amount_result_30d", "num_sessions_7d", "num_sessions_30d",
        "avg_sessions_duration_30d", "avg_bet_amount_30d",
        "net_game_result_7d", "net_game_result_30d",
        "failed_withdrawals_30d", "deposit_count_30d", "withdrawal_count_30d",
        "withdrawal_ratio"
    ]
    categorical_cols = ["country", "age_bucket"]
    
    expanded_features = []
    expanded_features.extend(numeric_cols)
    
    for input_col, category_sizes in zip(categorical_cols, ohe_model.categorySizes):
        for i in range(category_sizes):
            expanded_features.append(f"{input_col}_{i}")

    fi_df = pd.DataFrame({"feature": expanded_features, "coefficient": coeffs})
    mlflow.log_table(fi_df, "feature_importance.json")
    logger.info(f"Logged feature importance for {len(expanded_features)} features")


def _get_git_info() -> Tuple[str, str]:
    """
    Extract git commit and branch information.
    
    Returns:
        Tuple of (commit_hash, branch_name)
    """
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
        ).decode().strip()
    except Exception as e:
        logger.warning(f"Could not get git commit: {e}")
        git_commit = "unknown"

    try:
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ).decode().strip()
    except Exception as e:
        logger.warning(f"Could not get git branch: {e}")
        git_branch = "unknown"

    return git_commit, git_branch


def _train_final_models(
    pipeline_template: Pipeline,
    train_val_df: DataFrame,
    test_df: DataFrame,
    evaluator: BinaryClassificationEvaluator,
    best_reg: float,
    best_elastic: float,
    thresholds_to_test: List[float],
    numeric_cols: List[str],
    categorical_cols: List[str],
    parent_run_id: str = None
) -> None:
    """
    Train final models on train+val set using selected thresholds, evaluate on test set.
    
    For each threshold:
    1. Create new pipeline with fixed hyperparams + threshold
    2. Train on train+val data
    3. Log model to MLflow registry
    4. Evaluate on test set
    5. Log test metrics and confusion matrix
    
    Args:
        pipeline_template: Template pipeline (for extracting stages)
        train_val_df: Combined train+val data with class weights
        test_df: Test data with class weights
        evaluator: AUPR evaluator
        best_reg: Best regularization parameter from CV
        best_elastic: Best elasticNet parameter from CV
        thresholds_to_test: List of thresholds to train models for
        numeric_cols: Numeric feature names
        categorical_cols: Categorical feature names
        parent_run_id: Optional parent run ID for nested run tracking
    """
    logger.info(f"Training final models for {len(thresholds_to_test)} thresholds...")
    
    categorical_idx = [c + "_idx" for c in categorical_cols]
    categorical_ohe = [c + "_ohe" for c in categorical_cols]

    for th in thresholds_to_test:
        logger.info(f"Training final model with threshold={th:.2f}...")
        
        try:
            # Create nested parent run for this threshold if parent_run_id provided
            is_nested = parent_run_id is not None
            with mlflow.start_run(run_name=f'final_training_threshold_{th:.2f}', nested=is_nested) as train_run:
                # Link to parent run if provided
                if is_nested:
                    mlflow.set_tag("parent_run_id", parent_run_id)
                    mlflow.set_tag("phase", "final_training")
                    mlflow.set_tag("threshold_group", f"{th:.2f}")
                
                # Log train+test data ranges
                train_dates = train_val_df.agg(F.min("reference_date"), F.max("reference_date")).first()
                test_dates = test_df.agg(F.min("reference_date"), F.max("reference_date")).first()
                
                mlflow.log_param("train_start", str(train_dates[0]))
                mlflow.log_param("train_end", str(train_dates[1]))
                mlflow.log_param("test_start", str(test_dates[0]))
                mlflow.log_param("test_end", str(test_dates[1]))
                mlflow.log_param("threshold", th)
                mlflow.log_param("regParam", best_reg)
                mlflow.log_param("elasticNetParam", best_elastic)

                # System info
                mlflow.log_param("python_version", sys.version.split()[0])
                mlflow.log_param("platform", platform.platform())
                git_commit, git_branch = _get_git_info()
                mlflow.log_param("git_commit", git_commit)
                mlflow.log_param("git_branch", git_branch)

                # Data info
                num_churn = train_val_df.filter("next_7d_churn = true").count()
                num_nonchurn = train_val_df.filter("next_7d_churn = false").count()
                class_weight = num_nonchurn / num_churn if num_churn > 0 else 1.0
                
                mlflow.log_param("train_rows", train_val_df.count())
                mlflow.log_param("test_rows", test_df.count())
                mlflow.log_param("num_churn_train", num_churn)
                mlflow.log_param("num_nonchurn_train", num_nonchurn)
                mlflow.log_param("class_weight", class_weight)
                mlflow.log_param("num_features", len(numeric_cols) + len(categorical_ohe))

                # Build and train final pipeline
                indexer = StringIndexer(inputCols=categorical_cols, outputCols=categorical_idx, handleInvalid="error")
                ohe = OneHotEncoder(inputCols=categorical_idx, outputCols=categorical_ohe, dropLast=False)
                numeric_assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
                scaler = StandardScaler(inputCol="numeric_features", outputCol="numeric_features_scaled", withMean=True, withStd=True)
                final_assembler = VectorAssembler(inputCols=["numeric_features_scaled"] + categorical_ohe, outputCol="features")

                lr_final = LogisticRegression(
                    featuresCol="features",
                    labelCol="next_7d_churn_idx",
                    weightCol="class_weight",
                    regParam=best_reg,
                    elasticNetParam=best_elastic,
                    maxIter=50,
                    threshold=th
                )

                final_pipeline = Pipeline(stages=[indexer, ohe, numeric_assembler, scaler, final_assembler, lr_final])
                final_model = final_pipeline.fit(train_val_df)

                # Log feature importance
                _log_feature_importance(final_model, train_val_df)

                # Register model
                sample_input = train_val_df.limit(100).toPandas()
                train_preds = final_model.transform(train_val_df).withColumn("p_churn", vector_to_array("probability")[1])
                sample_output = train_preds.select("p_churn").limit(100).toPandas()
                signature = infer_signature(sample_input, sample_output)

                mlflow.spark.log_model(
                    spark_model=final_model,
                    artifact_path='spark_model',
                    registered_model_name='SparkLogisticRegression_train',
                    signature=signature,
                    metadata={"threshold": th}
                )

                logger.info(f"Registered final model for threshold {th:.2f}")
                final_run_id = train_run.info.run_id

                # Evaluate on test set (nested child run within threshold group)
                logger.info(f"Evaluating on test set...")
                with mlflow.start_run(run_name=f'evaluation_threshold_{th:.2f}', nested=True):
                    if is_nested:
                        mlflow.set_tag("parent_run_id", parent_run_id)
                    mlflow.set_tag("parent_threshold_run", final_run_id)
                    mlflow.set_tag("phase", "evaluation")
                    
                    final_model = mlflow.spark.load_model(f"runs:/{final_run_id}/spark_model")
                    
                    test_preds = final_model.transform(test_df).withColumn("p_churn", vector_to_array("probability")[1])
                    test_aupr = evaluator.evaluate(test_preds)

                    precision, recall, f1, _, _ = compute_metrics(test_preds, th)
                    
                    mlflow.log_param('threshold', th)
                    mlflow.log_metric('f1', f1)
                    mlflow.log_metric('recall', recall)
                    mlflow.log_metric('precision', precision)
                    mlflow.log_metric("test_aupr", test_aupr)

                    # Confusion matrix
                    cm = (
                        test_preds
                        .withColumn("pred_label", (F.col("p_churn") >= th).cast("int"))
                        .groupBy("next_7d_churn_idx", "pred_label")
                        .count()
                    )
                    mlflow.log_table(cm.toPandas(), "confusion_matrix.json")
                    
                    logger.info(f"Test metrics (threshold {th:.2f}): precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f}, AUPR={test_aupr:.3f}")

        except Exception as e:
            logger.error(f"Failed to train/evaluate model with threshold {th:.2f}: {e}")
            continue


def main() -> None:
    """
    Main training pipeline orchestration with MLflow parent-child hierarchy.
    
    Workflow:
    1. Load and split data chronologically
    2. Compute class weights for imbalance handling
    3. Build Spark ML pipeline (feature engineering + LR)
    4. Perform 3-fold cross-validation for hyperparameter tuning
    5. Optimize decision threshold on validation set
    6. Train final models on train+val for selected thresholds
    7. Evaluate all models on test set
    
    MLflow Structure (with parent-child hierarchy):
    ├── Experiment: "churn_prediction_v2"
    │   ├── Parent Run: "00_hyperparameter_tuning"
    │   │   └── CV training metrics and best model registration
    │   ├── Parent Run: "01_threshold_optimization"  
    │   │   └── Threshold candidate evaluation and PR curve
    │   └── Parent Run: "02_final_training"
    │       └── Per-threshold model training
    │           ├── Child Run: "final_training_threshold_X.XX"
    │           │   └── Child Run: "evaluation_threshold_X.XX"
    │           ├── Child Run: "final_training_threshold_Y.YY"
    │           │   └── Child Run: "evaluation_threshold_Y.YY"
    │           ...
    
    Raises:
        Exception: If any major pipeline step fails
    """
    logger.info("=" * 80)
    logger.info("STARTING LOGISTIC REGRESSION MODEL TRAINING WITH MLflow HIERARCHY")
    logger.info("=" * 80)

    spark = get_spark()
    spark.catalog.clearCache()
    config_ = DataGenConfig()

    # Setup MLflow with experiment and tags
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("churn_prediction_v2")
    
    # Log experiment-level metadata
    experiment_metadata = {
        "data_scope": "full_dataset",
        "model_type": "logistic_regression_churn_prediction",
        "pipeline_version": "2.1_hierarchical_mlflow",
        "experiment_timestamp": datetime.now().isoformat(),
        "architecture": "spark_ml_pipeline",
    }
    mlflow.set_tags(experiment_metadata)
    
    try:
        # Create master run context for entire experiment
        with mlflow.start_run(run_name="00_hyperparameter_tuning") as master_run:
            master_run_id = master_run.info.run_id
            logger.info(f"Master experiment run ID: {master_run_id}")
            
            # Step 1: Prepare data (loads Gold tables internally)
            logger.info("Step 1/6: Preparing data...")
            train_df, val_df, test_df, numeric_cols, categorical_cols = _prepare_data(spark, sample_fraction=1.0)

            # Step 2: Compute class weights
            logger.info("Step 2/6: Computing class weights...")
            num_churn = train_df.filter("next_7d_churn = true").count()
            num_nonchurn = train_df.filter("next_7d_churn = false").count()
            weight_for_churn = num_nonchurn / num_churn if num_churn > 0 else 1.0
            logger.info(f"Class weight for churn: {weight_for_churn:.2f} (churn={num_churn}, non-churn={num_nonchurn})")

            # Log to master run
            mlflow.log_param("train_rows", train_df.count())
            mlflow.log_param("val_rows", val_df.count())
            mlflow.log_param("test_rows", test_df.count())
            mlflow.log_param("num_churn_total", num_churn)
            mlflow.log_param("num_nonchurn_total", num_nonchurn)
            mlflow.log_param("numeric_features", len(numeric_cols))
            mlflow.log_param("categorical_features", len(categorical_cols))

            # Apply weights
            train_df = add_class_weight(train_df, weight_for_churn)
            val_df = add_class_weight(val_df, weight_for_churn)
            test_df = add_class_weight(test_df, weight_for_churn)

            # Train+val combined
            num_churn_train_val = (train_df.filter("next_7d_churn = true").count() + 
                                   val_df.filter("next_7d_churn = true").count())
            num_nonchurn_train_val = (train_df.filter("next_7d_churn = false").count() + 
                                      val_df.filter("next_7d_churn = false").count())
            weight_for_churn_train_val = num_nonchurn_train_val / num_churn_train_val if num_churn_train_val > 0 else 1.0
            
            train_val_df = train_df.unionByName(val_df)
            train_val_df = add_class_weight(train_val_df, weight_for_churn_train_val)
            test_df = add_class_weight(test_df, weight_for_churn_train_val)

            # Step 3: Build pipeline
            logger.info("Step 3/6: Building Spark ML pipeline...")
            pipeline, evaluator = _build_pipeline(numeric_cols, categorical_cols)

            # Step 4: Hyperparameter tuning (nested under master run)
            logger.info("Step 4/6: Starting hyperparameter tuning...")
            best_model, best_reg, best_elastic, cv_run_id = _tune_hyperparams(
                pipeline, train_df, evaluator, parent_run_id=master_run_id
            )

        # Step 5: Threshold optimization (separate parent run)
        logger.info("Step 5/6: Starting threshold optimization...")
        with mlflow.start_run(run_name="01_threshold_optimization") as threshold_run:
            threshold_run_id = threshold_run.info.run_id
            mlflow.set_tag("experiment_master_run", master_run_id)
            
            th_f1, th_rec_f1_05, th_rec_08, th_flagged_players, _ = _optimize_threshold(
                best_model, val_df, numeric_cols, categorical_cols, parent_run_id=threshold_run_id
            )

        # Step 6: Train final models (separate parent run with nested threshold groups)
        logger.info("Step 6/6: Starting final model training...")
        with mlflow.start_run(run_name="02_final_training") as final_training_run:
            final_training_run_id = final_training_run.info.run_id
            mlflow.set_tag("hyperparameter_tuning_run", master_run_id)
            mlflow.set_tag("threshold_optimization_run", threshold_run_id)
            mlflow.log_param("num_threshold_models", len([th_f1, th_rec_f1_05, th_rec_08, th_flagged_players]))
            
            thresholds_to_test = [th_f1, th_rec_f1_05, th_rec_08, th_flagged_players]
            _train_final_models(
                pipeline, train_val_df, test_df, evaluator,
                best_reg, best_elastic, thresholds_to_test,
                numeric_cols, categorical_cols,
                parent_run_id=final_training_run_id
            )

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"Master Run ID: {master_run_id}")
        logger.info(f"Threshold Optimization Run ID: {threshold_run_id}")
        logger.info(f"Final Training Run ID: {final_training_run_id}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()















