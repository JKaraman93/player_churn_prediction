"""
Constants Module: Project-wide Configuration Constants

Centralizes all magic numbers, default values, and configuration constants
to improve maintainability and enable easy adjustments.
"""

# Data paths
BRONZE_DATA_PATH = "./data/bronze"
SILVER_DATA_PATH = "./data/silver"
GOLD_DATA_PATH = "./data/gold"

# Feature engineering windows (days)
ROLLING_WINDOW_7_DAYS = 7
ROLLING_WINDOW_30_DAYS = 30

# Churn definition
CHURN_INACTIVITY_DAYS = 7
MIN_HISTORY_DAYS = 30

# Data quality thresholds
MIN_PLAYERS_FOR_TRAINING = 100
MIN_DATA_POINTS_PER_FEATURE = 10

# Model training
RANDOM_SEED = 42
TRAIN_TEST_SPLIT_RATIO = 0.7
VALIDATION_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15

# Class imbalance handling
#POS_LABEL_WEIGHT = 3.0  # Weight for positive (churn) class
#NEG_LABEL_WEIGHT = 1.0  # Weight for negative (non-churn) class

# Evaluation metrics
PRIMARY_METRIC = "AUPR"  # Area Under Precision-Recall Curve
MIN_ACCEPTABLE_AUPR_THRESHOLD = 0.5

# Risk segmentation thresholds
LOW_RISK_THRESHOLD = 0.25
MEDIUM_RISK_THRESHOLD = 0.50
HIGH_RISK_THRESHOLD = 0.75

# Logging and monitoring
LOG_LEVEL = "INFO"
MLFLOW_EXPERIMENT_NAME = "churn_prediction"

# Spark configuration
SPARK_DRIVER_MEMORY = "12g"
SPARK_EXECUTOR_MEMORY = "12g"
SPARK_MAX_PARTITIONS = 200

# File I/O
OUTPUT_FORMAT = "parquet"
PARQUET_COMPRESSION = "snappy"
