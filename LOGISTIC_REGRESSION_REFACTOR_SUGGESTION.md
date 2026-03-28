# Logistic Regression Refactor Suggestion: MLflow-Aware Architecture

## Current Issues with Proposed Refactor

The current helper function structure is good, but to truly leverage MLflow while maintaining clean code, consider this architecture:

### Problem 1: MLflow Runs Scattered in Helper Functions
Current flow:
```
main()
├── _tune_hyperparams() ← Creates run 'train'
├── _optimize_threshold() ← Creates run 'threshold_optimization'  
└── _train_final_models() ← Creates runs 'final_train_*' and 'final_test_*'
```

**Issue**: No parent-child relationship between runs. Hard to track experiment hierarchy.

---

## Proposed Refactor: Parent-Child MLflow Hierarchy

### Architecture with Nested Runs

```
main() [Parent Context - Sets experiment]
├── CV Training Phase [Parent Run: 'hyperparameter_tuning']
│   └── _tune_hyperparams() [Logs to parent run]
│
├── Threshold Optimization Phase [Parent Run: 'threshold_selection']
│   └── _optimize_threshold() [Logs to parent run]
│
└── Final Training & Evaluation Phase [Parent Run: 'final_training']
    └── _train_final_models() [Creates child runs per threshold]
        ├── child_run: "model_train_thres_0.45"
        ├── child_run: "model_eval_thres_0.45"
        ├── child_run: "model_train_thres_0.50"
        └── child_run: "model_eval_thres_0.50"
```

---

## Implementation Strategy (Keep All MLflow Features)

### Key Principle
**Do NOT remove MLflows runs/logging - wrap them in a parent context and add nested runs**

### Step 1: Modify `main()` to Set Experiment Context

```python
def main() -> None:
    """Main training pipeline with MLflow experiment hierarchy."""
    logger.info("=" * 80)
    logger.info("STARTING LOGISTIC REGRESSION MODEL TRAINING")
    logger.info("=" * 80)

    spark = get_spark()
    spark.catalog.clearCache()
    config_ = DataGenConfig()

    # Setup MLflow EXPERIMENT (not run - runs go inside)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("churn_prediction_v2")
    
    experiment_tags = {
        "data_scope": "full_dataset",
        "model_type": "logistic_regression",
        "pipeline_version": "2.0_production"
    }
    mlflow.set_tags(experiment_tags)

    try:
        # PARENT CONTEXT: CV Training
        with mlflow.start_run(run_name="00_hyperparameter_tuning") as cv_parent_run:
            mlflow.log_param("phase", "hyperparameter_tuning")
            
            # Load data and prepare
            train_df, val_df, test_df, numeric_cols, categorical_cols = _prepare_data(spark)
            
            # Compute class weights
            num_churn = train_df.filter("next_7d_churn = true").count()
            num_nonchurn = train_df.filter("next_7d_churn = false").count()
            weight_for_churn = num_nonchurn / num_churn if num_churn > 0 else 1.0
            
            # [Log shared parameters to parent]
            mlflow.log_param("train_rows", train_df.count())
            mlflow.log_param("val_rows", val_df.count())
            mlflow.log_param("test_rows", test_df.count())
            mlflow.log_param("class_weight", weight_for_churn)
            
            # Apply weights
            train_df = add_class_weight(train_df, weight_for_churn)
            val_df = add_class_weight(val_df, weight_for_churn)
            test_df = add_class_weight(test_df, weight_for_churn)
            
            # Build pipeline
            pipeline, evaluator = _build_pipeline(numeric_cols, categorical_cols)
            
            # Tune hyperparameters WITHIN parent context
            best_model, best_reg, best_elastic, _ = _tune_hyperparams(
                pipeline, train_df, evaluator, 
                parent_run_id=cv_parent_run.info.run_id  # PASS parent run ID
            )

        # PARENT CONTEXT: Threshold Optimization
        with mlflow.start_run(run_name="01_threshold_optimization") as threshold_parent_run:
            mlflow.log_param("phase", "threshold_selection")
            mlflow.log_param("regParam", best_reg)
            mlflow.log_param("elasticNetParam", best_elastic)
            
            th_f1, th_rec_f1_05, th_rec_08, th_flagged, _ = _optimize_threshold(
                best_model, val_df, numeric_cols, categorical_cols,
                parent_run_id=threshold_parent_run.info.run_id  # PASS parent run ID
            )

        # PARENT CONTEXT: Final Training & Evaluation
        with mlflow.start_run(run_name="02_final_training_evaluation") as final_parent_run:
            mlflow.log_param("phase", "final_training")
            mlflow.log_param("regParam", best_reg)
            mlflow.log_param("elasticNetParam", best_elastic)
            mlflow.log_param("num_thresholds", 4)
            
            thresholds_to_test = [th_f1, th_rec_f1_05, th_rec_08, th_flagged]
            _train_final_models(
                pipeline, train_val_df, test_df, evaluator,
                best_reg, best_elastic, thresholds_to_test,
                numeric_cols, categorical_cols,
                parent_run_id=final_parent_run.info.run_id  # PASS parent run ID
            )

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        raise
```

### Step 2: Modify Helper Functions to Accept Parent Run ID

**For `_tune_hyperparams()`:**
```python
def _tune_hyperparams(
    pipeline: Pipeline, 
    train_df: DataFrame, 
    evaluator: BinaryClassificationEvaluator,
    parent_run_id: str = None  # NEW PARAMETER
) -> Tuple[...]:
    """
    Perform grid search cross-validation.
    
    Args:
        pipeline: Spark ML pipeline
        train_df: Training data
        evaluator: AUPR evaluator
        parent_run_id: Parent MLflow run ID (optional)
        
    Returns:
        Tuple of (best_model, best_reg, best_elastic, run_id)
    """
    logger.info("Starting hyperparameter tuning...")
    
    paramGrid = ParamGridBuilder() \
        .addGrid(pipeline.getStages()[-1].regParam, [0.01, 0.1, 0.5]) \
        .addGrid(pipeline.getStages()[-1].elasticNetParam, [0.0, 0.5]) \
        .build()

    cv = CrossValidator(...)

    try:
        # NESTED RUN: Still enable nested runs if parent exists
        with mlflow.start_run(run_name='cv_training', nested=parent_run_id is not None) as cv_run:
            if parent_run_id:
                mlflow.set_tag("parent_run_id", parent_run_id)
            
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
            train_preds = best_model.transform(train_df).withColumn("p_churn", ...)
            train_aupr = evaluator.evaluate(train_preds)
            mlflow.log_metric("train_aupr", train_aupr)
            
            _log_feature_importance(best_model, train_df)

            run_id = cv_run.info.run_id

        return best_model, best_reg, best_elastic, run_id

    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        raise
```

### Step 3: Modify Other Helper Functions Similarly

**For `_optimize_threshold()`:**
```python
def _optimize_threshold(
    best_model: PipelineModel, 
    val_df: DataFrame, 
    numeric_cols: List[str], 
    categorical_cols: List[str],
    parent_run_id: str = None  # NEW PARAMETER
) -> Tuple[...]:
    """
    Find optimal decision threshold using validation set.
    
    Args:
        parent_run_id: Parent MLflow run ID (optional)
        ...
    """
    logger.info("Optimizing decision threshold...")
    
    thresholds = [i / 100 for i in range(5, 96, 5)]
    
    try:
        # NESTED RUN
        with mlflow.start_run(run_name='threshold_testing', nested=parent_run_id is not None) as opt_run:
            if parent_run_id:
                mlflow.set_tag("parent_run_id", parent_run_id)
            
            results = []
            val_preds = best_model.transform(val_df).withColumn("p_churn", ...)
            val_preds.persist()
            
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
            
            # Plot PR curve
            pdf = val_preds.select("p_churn", "next_7d_churn_idx").toPandas()
            precision_arr, recall_arr, _ = precision_recall_curve(pdf["next_7d_churn_idx"], pdf["p_churn"])
            plt.figure()
            plt.plot(recall_arr, precision_arr)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve (Validation)")
            mlflow.log_figure(plt.gcf(), "pr_curve.png")
            plt.close()

            val_preds.unpersist()
            
            # Select thresholds
            metrics_df = get_spark().createDataFrame(results)
            th_f1 = metrics_df.orderBy(F.desc("f1")).first()["threshold"]
            # ... other threshold selections

        return th_f1, th_rec_f1_05, th_rec_08, th_flagged_players, opt_run.info.run_id

    except Exception as e:
        logger.error(f"Threshold optimization failed: {e}")
        raise
```

**For `_train_final_models()`:**
```python
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
    parent_run_id: str = None  # NEW PARAMETER
) -> None:
    """
    Train final models on train+val, evaluate on test set.
    
    Args:
        parent_run_id: Parent MLflow run ID (optional)
        ...
    """
    logger.info(f"Training {len(thresholds_to_test)} final models...")
    
    for th in thresholds_to_test:
        logger.info(f"Training final model with threshold={th:.2f}...")
        
        try:
            # NESTED RUN per threshold
            with mlflow.start_run(run_name=f'model_train_thres_{th:.2f}', nested=parent_run_id is not None):
                if parent_run_id:
                    mlflow.set_tag("parent_run_id", parent_run_id)
                
                # ... training code ...
                mlflow.log_param("threshold", th)
                # ... log metrics and model ...
                
                final_run_id = mlflow.active_run().info.run_id

        except Exception as e:
            logger.error(f"Failed to train model with threshold {th:.2f}: {e}")
            continue

        # NESTED RUN for evaluation
        try:
            with mlflow.start_run(run_name=f'model_eval_thres_{th:.2f}', nested=parent_run_id is not None):
                if parent_run_id:
                    mlflow.set_tag("parent_run_id", parent_run_id)
                    mlflow.set_tag("train_run_id", final_run_id)
                
                # ... evaluation code ...
                mlflow.log_metric('f1', f1)
                mlflow.log_metric('recall', recall)
                mlflow.log_metric('precision', precision)
                # ... log confusion matrix ...

        except Exception as e:
            logger.error(f"Failed to evaluate threshold {th:.2f}: {e}")
            continue
```

---

## Benefits of This Refactor

✅ **Preserves All MLflow Features**:
- All existing parameter logging remains
- All existing metric logging remains
- All existing artifact logging remains
- All existing run tracking remains

✅ **Adds Clear Hierarchy**:
- Parent-child run relationships visible in MLflow UI
- Easy to track which thresholds belong to which training phase
- Clear experiment structure

✅ **Maintains Readability**:
- Helper functions stay focused and clean
- Parent context management stays in `main()`
- No code duplication

✅ **Better Debugging**:
- Tags link child runs to parents
- Parent run shows aggregate metrics
- Easier to reproduce specific training phases

---

## MLflow UI Result

After this refactor, MLflow UI will show:
```
Experiment: churn_prediction_v2
├── Run: 00_hyperparameter_tuning (parent run)
│   ├── Params: cv_folds, cv_parallelism, regParam, elasticNetParam
│   ├── Metrics: train_aupr, other metrics
│   └── Artifacts: feature_importance.json, signature
│
├── Run: 01_threshold_optimization (parent run)
│   ├── Params: regParam, elasticNetParam, threshold values
│   ├── Metrics: precision/recall for each threshold
│   └── Artifacts: threshold_metrics.json, pr_curve.png
│
└── Run: 02_final_training_evaluation (parent run)
    ├── Run: model_train_thres_0.45 (nested)
    ├── Run: model_eval_thres_0.45 (nested)
    ├── Run: model_train_thres_0.50 (nested)
    └── Run: model_eval_thres_0.50 (nested)
```

---

## Implementation Notes

1. **No Breaking Changes**: All existing MLflow functionality is preserved
2. **Backward Compatible**: Can still access all runs/metrics the same way
3. **Gradual Migration**: Can be implemented incrementally
4. **Error Handling**: All try/except blocks preserved
5. **Logging**: All logger calls preserved and enhanced with context

---

## To Implement

1. Modify `main()` to add parent MLflow contexts (**~30 lines**)
2. Add `parent_run_id` parameter to helper functions (**~5 lines each**)
3. Add nested run support in helper functions (**~3 lines each**)
4. Add tags linking parent-child runs (**~2 lines each**)

**Total: ~50-60 lines of changes across the file**

No existing code needs to be removed - only structured better!
