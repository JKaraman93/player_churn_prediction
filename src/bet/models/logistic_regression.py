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

from bet.utils.spark_session import get_spark
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd
import bet.utils.config as config
from pyspark.ml.feature import StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import mlflow
import json
from mlflow.models.signature import infer_signature
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import platform
import subprocess
import numpy as np


def compute_metrics(df, threshold):
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

    select_cols = ['num_churned','num_flagged',]#'num_players']
    day_average = pred_per_day.select([F.round(F.avg(c).alias(c),0) for c in select_cols])
    day_avg_churned, day_avg_flagged = day_average.first()

    metrics = pred.groupBy("next_7d_churn_idx", "pred_label").count()
    tp = metrics.filter("next_7d_churn_idx = 1 AND pred_label = 1").select("count").first()
    fp = metrics.filter("next_7d_churn_idx = 0 AND pred_label = 1").select("count").first()
    fn = metrics.filter("next_7d_churn_idx = 1 AND pred_label = 0").select("count").first()

    tp = tp[0] if tp else 0
    fp = fp[0] if fp else 0
    fn = fn[0] if fn else 0

    num_rows = pred.count()
    num_flagged = tp + fp
    num_churned = tp + fn 

    precision = float(np.round(tp / (tp + fp) ,2)) if (tp + fp) > 0 else 0.0
    recall = float(np.round(tp / (tp + fn), 2)) if (tp + fn) > 0 else 0.0
    f1 = (float(np.round( 2 * precision * recall / (precision + recall) , 2))
        if (precision + recall) > 0 else 0.0
    )

    return precision, recall, f1 , day_avg_churned, day_avg_flagged

def add_class_weight(df, weight_for_churn):
    return df.withColumn(
        "class_weight",
        F.when(F.col("next_7d_churn"), weight_for_churn).otherwise(1.0)
    )


spark = get_spark()
spark.catalog.clearCache()
config_ = config.DataGenConfig()

player_behavior = spark.read.parquet("./data/gold/player_behavior")
player_snapshot = spark.read.parquet("./data/gold/player_snapshot")
labels = spark.read.parquet("./data/gold/labels")

sample_fraction = 1.0
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("second experiment")
experiment_tags = {
    "data_sample_fraction": sample_fraction, 
    "data_scope": "sampled",
    }

player_snapshot = player_snapshot.select(
'player_idx', 'country', 'age_bucket', 
#'device_type', #'acquisition_channel', #'registration_date', #'risk_segment',
  )

sample_players = player_snapshot.select("player_idx").sample(sample_fraction)
model_df = (
    player_behavior
        .join(player_snapshot, on="player_idx", how="left")
        .join(labels, on=["player_idx", "reference_date"], how="inner")
)

sample_dataset = sample_players.join(model_df, on="player_idx", how="inner") \
                               .withColumn("next_7d_churn_idx", F.col("next_7d_churn").cast("int"))


numeric_cols = [
    "balance_7d_ago", "balance_30d_ago", "net_amount_result_7d",
    "net_amount_result_30d", "num_sessions_7d", "num_sessions_30d",
    "avg_sessions_duration_30d", "avg_bet_amount_30d",
    "net_game_result_7d", "net_game_result_30d",
    "failed_withdrawals_30d", "deposit_count_30d", "withdrawal_count_30d",
    "withdrawal_ratio"
]

categorical_cols = ["country", "age_bucket"]
categorical_idx = [c + "_idx" for c in categorical_cols]
categorical_ohe = [c + "_ohe" for c in categorical_cols]

indexer = StringIndexer(inputCols=categorical_cols, outputCols=categorical_idx, handleInvalid="error")
ohe = OneHotEncoder(inputCols=categorical_idx, outputCols=categorical_ohe, dropLast=False)

numeric_assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
scaler = StandardScaler(inputCol="numeric_features", outputCol="numeric_features_scaled", withMean=True, withStd=True)

final_assembler = VectorAssembler(inputCols=["numeric_features_scaled"] + categorical_ohe, outputCol="features")

lr = LogisticRegression(featuresCol="features", labelCol="next_7d_churn_idx", weightCol="class_weight", maxIter=50)

pipeline = Pipeline(stages=[indexer, ohe, numeric_assembler, scaler, final_assembler, lr])

evaluator = BinaryClassificationEvaluator(labelCol="next_7d_churn_idx", metricName="areaUnderPR")

paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
    .build()

cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3, parallelism=2)

feature_metadata = {
    "numeric_features": numeric_cols,
    "categorical_features": categorical_cols,
    "label": "next_7d_churn",
    "label_definition": "completion of 7-day inactivity window within next 7 days"
}


# ---------------------------
# Train / Val / Test Split
# ---------------------------

dates = [row.reference_date for row in sample_dataset.select("reference_date").distinct().orderBy("reference_date").collect()]
n = len(dates)
train_cut = dates[int(n * 0.70)]
val_cut = dates[int(n * 0.85)]

train_df = sample_dataset.filter(F.col("reference_date") < train_cut)
val_df = sample_dataset.filter((F.col("reference_date") >= train_cut) & (F.col("reference_date") < val_cut))
test_df = sample_dataset.filter(F.col("reference_date") >= val_cut)
train_val_df = train_df.unionByName(val_df)

# Class weighting
num_churn = train_df.filter("next_7d_churn = true").count()
num_nonchurn = train_df.filter("next_7d_churn = false").count()
weight_for_churn = num_nonchurn / num_churn

train_df = add_class_weight(train_df, weight_for_churn)
val_df = add_class_weight(val_df, weight_for_churn)
test_df = add_class_weight(test_df, weight_for_churn)

# ---------------------------
# Train+Val / Test Split
# ---------------------------

train_val_df = train_df.unionByName(val_df)

# Class weighting
num_churn_train_val = train_val_df.filter("next_7d_churn = true").count()
num_nonchurn_train_val = train_val_df.filter("next_7d_churn = false").count()
weight_for_churn_train_val = num_nonchurn_train_val / num_churn_train_val

train_val_df = add_class_weight(train_val_df, weight_for_churn_train_val)
test_df = add_class_weight(test_df, weight_for_churn_train_val)

########## training set ##################
#### hyperparameter tuning #########

with mlflow.start_run(run_name='train') as run:
    mlflow.log_dict(experiment_tags, "experiment_tags.json")

    mlflow.log_param('sample_of_players',str(sample_fraction*100) + '%')

    # ---------------- DATA METADATA ----------------
    mlflow.log_param("train_start", str(train_df.agg(F.min("reference_date")).first()[0]))
    mlflow.log_param("train_end",   str(train_df.agg(F.max("reference_date")).first()[0]))
    mlflow.log_param("val_start",   str(val_df.agg(F.min("reference_date")).first()[0]))
    mlflow.log_param("val_end",     str(val_df.agg(F.max("reference_date")).first()[0]))
    mlflow.log_param("test_start",  str(test_df.agg(F.min("reference_date")).first()[0]))
    mlflow.log_param("test_end",    str(test_df.agg(F.max("reference_date")).first()[0]))

    mlflow.log_param("python_version", sys.version)
    mlflow.log_param("platform", platform.platform())


    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
        ).decode().strip()
    except Exception as e:
        git_commit = "unknown"
        git_error = str(e)

    mlflow.log_param("git_commit", git_commit)

    try:
        git_branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ).decode().strip()
    except Exception:
        git_branch = "unknown"

    mlflow.log_param("git_branch", git_branch)


    mlflow.log_param("train_rows", train_df.count())
    mlflow.log_param("val_rows", val_df.count())
    mlflow.log_param("test_rows", test_df.count())

    mlflow.log_param("num_features", len(numeric_cols) + len(categorical_ohe))

    mlflow.log_param("num_churn_train", num_churn)
    mlflow.log_param("num_nonchurn_train", num_nonchurn)
    mlflow.log_param("class_weight", weight_for_churn)

    cv_model = cv.fit(train_df)
    best_model = cv_model.bestModel
    lr_best_model = best_model.stages[-1]
    best_reg = lr_best_model.getRegParam()
    best_elastic = lr_best_model.getElasticNetParam()

    mlflow.log_param("elasticParam", best_elastic)
    mlflow.log_param("regParam", best_reg)

    train_preds = best_model.transform(train_df).withColumn("p_churn", vector_to_array("probability")[1])
    train_upr = evaluator.evaluate(train_preds)
    mlflow.log_metric("upr", train_upr)

    coeffs = lr_best_model.coefficients.toArray().tolist()
    feature_names = numeric_cols + categorical_ohe
    # Suppose you have the fitted OHE stage in the pipeline
    ohe_model = best_model.stages[1]  # adjust index for your pipeline

    expanded_features = []

    # Add numeric features first
    expanded_features.extend(numeric_cols)

    # For each categorical column
    for input_col, output_col, category_sizes in zip(categorical_cols, categorical_ohe, ohe_model.categorySizes):
        for i in range(category_sizes):
            expanded_features.append(f"{input_col}_{i}")  # e.g., country_0, country_1, ...


    fi = pd.DataFrame({ "feature": expanded_features,  "coefficient": coeffs  })
    mlflow.log_table(fi, "feature_importance.json")


    sample_input = train_df.limit(100).toPandas()
    sample_output = train_preds.select("p_churn").limit(100).toPandas()
    signature = infer_signature(sample_input, sample_output)

    mlflow.spark.log_model(
        spark_model=best_model,
        artifact_path='spark_model',
        registered_model_name='SparkLogisticRegression_initial_train',
        signature=signature,
    )

    train_run_id = run.info.run_id

model_uri = f"runs:/{train_run_id}/spark_model"
loaded_model = mlflow.spark.load_model(model_uri)

thresholds = [i / 100 for i in range(5, 96, 5)]


##### find optimal threshold test on validation set ####
with mlflow.start_run(run_name='thresholds'):
    mlflow.set_tag("train_run_id", train_run_id)
    results = []

    val_preds = loaded_model.transform(val_df).withColumn("p_churn", vector_to_array("probability")[1])
    val_preds.persist()
    val_preds.count()
    
    for t in thresholds:
        #with mlflow.start_run(nested=True, run_name=f"thr_{t}"):
        precision, recall, f1, day_avg_churned, day_avg_flagged = compute_metrics(val_preds, t)
        results.append({'threshold': t, 'precision':precision, 'recall':recall, 'f1':f1,  
        'day_avg_churned': day_avg_churned, 'day_avg_flagged': day_avg_flagged})

    mlflow.log_table(pd.DataFrame(results), "threshold_metrics.json")    
    

    metrics_df = spark.createDataFrame(results)
    th_f1 = metrics_df.orderBy(F.desc("f1")).first()["threshold"]
    th_rec_f1_05 = metrics_df.filter(F.col('f1')>0.5).orderBy(F.desc("recall")).first()["threshold"]
    th_rec_08 = metrics_df.filter(F.col('recall')>0.8).orderBy(F.asc("recall")).first()["threshold"]
    th_flagged_players = (metrics_df.filter(F.col('day_avg_flagged')>F.col('day_avg_churned'))
    .orderBy(F.asc('day_avg_flagged'))).first()["threshold"]
    pdf = val_preds.select("p_churn", "next_7d_churn_idx").toPandas()

    precision_arr, recall_arr, _ = precision_recall_curve(
        pdf["next_7d_churn_idx"],
        pdf["p_churn"]
    )

    plt.figure()
    plt.plot(recall_arr, precision_arr)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    #plt.savefig("pr_curve.png")
    mlflow.log_figure(plt.gcf(), "pr_curve.png")
    plt.close()

val_preds.unpersist()


##### Train the model on train+val set ####
for th in [th_f1, th_rec_f1_05, th_rec_08, th_flagged_players ] :
    run_name_train = 'train+val_' + str(th)
    run_name_test = 'test_' + str(th)


    with mlflow.start_run(run_name=run_name_train) as run:

        mlflow.log_param("train_start", str(train_val_df.agg(F.min("reference_date")).first()[0]))
        mlflow.log_param("train_end",   str(train_val_df.agg(F.max("reference_date")).first()[0]))
        mlflow.log_param("test_start",  str(test_df.agg(F.min("reference_date")).first()[0]))
        mlflow.log_param("test_end",    str(test_df.agg(F.max("reference_date")).first()[0]))

        mlflow.log_param("python_version", sys.version)
        mlflow.log_param("platform", platform.platform())

        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
            ).decode().strip()
        except Exception as e:
            git_commit = "unknown"
            git_error = str(e)

        mlflow.log_param("git_commit", git_commit)

        try:
            git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
            ).decode().strip()
        except Exception:
            git_branch = "unknown"

        mlflow.log_param("git_branch", git_branch)


        mlflow.log_param("train_rows", train_val_df.count())
        mlflow.log_param("test_rows", test_df.count())

        mlflow.log_param("num_features", len(numeric_cols) + len(categorical_ohe))

        mlflow.log_param("num_churn_train", num_churn_train_val)
        mlflow.log_param("num_nonchurn_train", num_nonchurn_train_val)
        mlflow.log_param("class_weight", weight_for_churn_train_val)
        mlflow.log_param("elasticParam", best_elastic)
        mlflow.log_param("regParam", best_reg)
        mlflow.log_param('threshold', th)



        lr_final = LogisticRegression(
        featuresCol="features",
        labelCol="next_7d_churn_idx",
        weightCol="class_weight",
        regParam=best_reg,
        elasticNetParam=best_elastic,
        maxIter=50,
        threshold=th)

        final_pipeline = Pipeline(stages=[
            indexer, ohe,
            numeric_assembler,
            scaler,
            final_assembler,
            lr_final
        ])

        final_model = final_pipeline.fit(train_val_df)
        lr_model = final_model.stages[-1]
        train_val_preds = final_model.transform(train_val_df).withColumn("p_churn", vector_to_array("probability")[1])
        train_val_upr = evaluator.evaluate(train_val_preds)
        mlflow.log_metric("upr", train_upr)

        coeffs = lr_model.coefficients.toArray().tolist()
        feature_names = numeric_cols + categorical_ohe
        # Suppose you have the fitted OHE stage in the pipeline
        ohe_model = final_model.stages[1]  # adjust index for your pipeline

        expanded_features = []

        # Add numeric features first
        expanded_features.extend(numeric_cols)

        # For each categorical column
        for input_col, output_col, category_sizes in zip(categorical_cols, categorical_ohe, ohe_model.categorySizes):
            for i in range(category_sizes):
                expanded_features.append(f"{input_col}_{i}")  # e.g., country_0, country_1, ...


        fi = pd.DataFrame({ "feature": expanded_features,  "coefficient": coeffs  })
        mlflow.log_table(fi, "feature_importance.json")

        sample_input = train_df.limit(100).toPandas()
        sample_output = train_preds.select("p_churn").limit(100).toPandas()
        signature = infer_signature(sample_input, sample_output)

        mlflow.spark.log_model(
            spark_model=final_model,
            artifact_path='spark_model',
            registered_model_name='SparkLogisticRegression_train',
            signature=signature,
            metadata={"threshold": th}


        )

        final_run_id = run.info.run_id

    final_model_uri = f"runs:/{final_run_id}/spark_model"
    loaded_model = mlflow.spark.load_model(final_model_uri)

        

    with mlflow.start_run(run_name=run_name_test):
        
        mlflow.set_tag("train_run_id", final_run_id)
        test_preds = loaded_model.transform(test_df).withColumn("p_churn", vector_to_array("probability")[1])
        test_upr = evaluator.evaluate(test_preds)

        precision, recall, f1, day_avg_churned, day_avg_flagged = compute_metrics(test_preds, th)
        mlflow.log_param('threshold', th)
        mlflow.log_metric('f1', f1)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric("upr", test_upr)
        
        cm = (
        test_preds
        .withColumn("pred_label", (F.col("p_churn") >= th).cast("int"))
        .groupBy("next_7d_churn_idx", "pred_label")
        .count()
        )

        cm_pd = cm.toPandas()
        mlflow.log_table(cm_pd, "confusion_matrix.json")















