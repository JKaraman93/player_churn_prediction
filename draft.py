from src.utils.spark_session import get_spark
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import mlflow
import pandas as pd

# ---------------------------
# Helper functions
# ---------------------------

def compute_metrics(df, threshold, label_col="next_7d_churn_idx", pred_col="p_churn"):
    """
    Compute precision, recall, f1 for a Spark DataFrame at a given threshold.
    """
    pred = df.withColumn("pred_label", (F.col(pred_col) >= threshold).cast("int"))

    metrics = pred.groupBy(label_col, "pred_label").count()
    tp = metrics.filter(f"{label_col} = 1 AND pred_label = 1").select("count").first()
    fp = metrics.filter(f"{label_col} = 0 AND pred_label = 1").select("count").first()
    fn = metrics.filter(f"{label_col} = 1 AND pred_label = 0").select("count").first()

    tp = tp[0] if tp else 0
    fp = fp[0] if fp else 0
    fn = fn[0] if fn else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def add_class_weight(df, weight_for_churn):
    return df.withColumn(
        "class_weight",
        F.when(F.col("next_7d_churn"), weight_for_churn).otherwise(1.0)
    )

# ---------------------------
# Spark session & data
# ---------------------------

spark = get_spark()
spark.catalog.clearCache()

# Config
sample_fraction = 0.1
data_path = "./data/gold/"

# Read data
player_behavior = spark.read.parquet(f"{data_path}/player_behavior")
player_snapshot = spark.read.parquet(f"{data_path}/player_snapshot")
labels = spark.read.parquet(f"{data_path}/labels")

player_snapshot = player_snapshot.select(
'player_idx', 'country', 'age_bucket', 
#'device_type', #'acquisition_channel', #'registration_date', #'risk_segment',
  )
# Sample dataset
sample_players = player_snapshot.select("player_idx").sample(sample_fraction)
model_df = (
    player_behavior
        .join(player_snapshot, on="player_idx", how="left")
        .join(labels, on=["player_idx", "reference_date"], how="inner")
)

sample_dataset = sample_players.join(model_df, on="player_idx", how="inner") \
                               .withColumn("next_7d_churn_idx", F.col("next_7d_churn").cast("int"))

# ---------------------------
# Features
# ---------------------------

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

# Class weighting
num_churn = train_df.filter("next_7d_churn = true").count()
num_nonchurn = train_df.filter("next_7d_churn = false").count()
weight_for_churn = num_nonchurn / num_churn

train_df = add_class_weight(train_df, weight_for_churn)
val_df = add_class_weight(val_df, weight_for_churn)
test_df = add_class_weight(test_df, weight_for_churn)

# ---------------------------
# MLflow tracking
# ---------------------------

mlflow.set_experiment("first_experiment")
mlflow.set_experiment_tags({
    "data_sample_fraction": sample_fraction, 
    })

# ---------- TRAIN ----------
with mlflow.start_run(run_name="train") as run:
    mlflow.set_tag("data_sample_fraction", sample_fraction)
    mlflow.set_tag("data_scope", "sampled")
    mlflow.set_tag("dataset_version", "v1.3")

    cv_model = cv.fit(train_df)
    best_model = cv_model.bestModel
    lr_best_model = best_model.stages[-1]

    mlflow.log_param("elasticNetParam", lr_best_model.getElasticNetParam())
    mlflow.log_param("regParam", lr_best_model.getRegParam())

    train_preds = best_model.transform(train_df).withColumn("p_churn", vector_to_array("probability")[1])
    train_upr = evaluator.evaluate(train_preds)
    mlflow.log_metric("upr", train_upr)

    mlflow.spark.log_model(spark_model=best_model, artifact_path="spark_model", registered_model_name="SparkLogisticRegression")

    train_run_id = run.info.run_id

# Load trained model
model_uri = f"runs:/{train_run_id}/spark_model"
loaded_model = mlflow.spark.load_model(model_uri)

# ---------- VALIDATION ----------
with mlflow.start_run(run_name="val") as run:
    mlflow.set_tag("train_run_id", train_run_id)
    val_preds = loaded_model.transform(val_df).withColumn("p_churn", vector_to_array("probability")[1])
    val_upr = evaluator.evaluate(val_preds)
    mlflow.log_metric("upr", val_upr)

# ---------- THRESHOLD SWEEP ----------
thresholds = [i / 100 for i in range(5, 96, 5)]
val_preds.persist()
val_preds.count()  # materialize cache

with mlflow.start_run(run_name="thresholds") as run:
    mlflow.set_tag("train_run_id", train_run_id)
    results = []

    for t in thresholds:
        #with mlflow.start_run(nested=True, run_name=f"thr_{t}"):
        precision, recall, f1 = compute_metrics(val_preds, t)
        results.append((t, precision, recall, f1))
        # mlflow.log_param("threshold", t)
        # mlflow.log_metric("f1", f1)
        # mlflow.log_metric("recall", recall)
        # mlflow.log_metric("precision", precision)
    df_metrics = pd.DataFrame(results, columns=["threshold", "precision", "recall", "f1"])
    df_metrics.to_csv("threshold_metrics.csv", index=False, ) 
    mlflow.log_artifact("threshold_metrics.csv")

# Convert to Spark DataFrame
metrics_df = spark.createDataFrame(results, ["threshold", "precision", "recall", "f1"])
best_threshold = metrics_df.orderBy(F.desc("f1")).first()["threshold"]
val_preds.unpersist()

# ---------- TEST ----------
with mlflow.start_run(run_name="test") as run:
    mlflow.set_tag("train_run_id", train_run_id)
    test_preds = loaded_model.transform(test_df).withColumn("p_churn", vector_to_array("probability")[1])
    test_upr = evaluator.evaluate(test_preds)

    precision, recall, f1 = compute_metrics(test_preds, best_threshold)
    mlflow.log_param("threshold", best_threshold)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("upr", test_upr)
