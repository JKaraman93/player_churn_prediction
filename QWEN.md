# Player Churn Prediction — Project Context

## Overview

This is an end-to-end machine learning project that predicts player churn in a gaming platform. It simulates a real-world analytics workflow: synthetic data generation, medallion-style data pipelines (Bronze → Silver → Gold), time-aware feature engineering, model training with Spark ML, experiment tracking via MLflow, batch inference, and backtesting.

**Prediction unit:** `(player, reference_date) -> probability of churn completion in (t, t+7]`

The system produces daily risk scores, not one static score per player.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Primary language |
| PySpark 4.1.0 | Data processing & ML |
| Spark SQL / Spark ML | Feature engineering, model training |
| MLflow | Experiment tracking, model registry |
| scikit-learn | Precision-recall curves, threshold tuning |
| pandas / numpy | Local data manipulation |
| matplotlib | Visualization |
| Parquet | Storage format |

## Project Structure

```
src/bet/
├── ingestion/          # Synthetic data generation (players, sessions, transactions)
├── pipelines/          # Bronze → Silver → Gold dataset creation
├── models/             # Training (logistic_regression.py), inference, feature prep
├── evaluation/         # Backtesting and performance analysis
├── utils/              # Spark session, config, constants, data helpers, logging
└── schemas/            # Data schema documentation
```

### Key Files

| File | Purpose |
|---|---|
| `src/bet/pipelines/create_bronze_dataset.py` | Generates synthetic raw data (players, sessions, transactions) |
| `src/bet/pipelines/create_silver_dataset.py` | Cleans/deduplicates bronze data into trusted silver tables |
| `src/bet/pipelines/create_gold_dataset.py` | Builds ML-ready gold tables (features + labels) |
| `src/bet/models/logistic_regression.py` | Full training pipeline: CV, threshold optimization, MLflow logging |
| `src/bet/models/inference.py` | Batch inference using registered production model |
| `src/bet/models/prepare_data_inference.py` | Feature recomputation for inference dates |
| `src/bet/evaluation/backtest.py` | Backtests model on held-out time periods |
| `src/bet/utils/config.py` | `DataGenConfig` dataclass for all pipeline parameters |
| `src/bet/utils/constants.py` | Project-wide constants (paths, thresholds, Spark config) |
| `src/bet/utils/spark_session.py` | `get_spark()` factory for configured SparkSession |
| `src/bet/utils/data_utils.py` | Helper functions for reading silver/gold tables |

## Data Architecture

### Bronze Layer — Raw synthetic data
- `players` — Player profiles with demographics, lifecycle stages, risk segments
- `sessions` — Gaming sessions with duration, bets, outcomes
- `transactions` — Financial transactions (deposits, withdrawals)

### Silver Layer — Cleaned, trusted source
- Deduplication, date consistency, balance validation
- Unified money-event construction

### Gold Layer — ML-ready tables
- `player_snapshot` — Static player attributes (country, age bucket)
- `player_behavior` — Rolling 7d/30d behavioral features
- `labels` — Future churn target aligned to each player-date

### Feature Examples
- Session counts (7d, 30d), avg session duration
- Avg bet amount, net game result
- Deposit/withdrawal counts, failed withdrawals, withdrawal ratio
- Balance snapshots (7d ago, 30d ago)

## Building and Running

### Prerequisites
- Python 3.10+
- Java runtime (for PySpark)
- Virtual environment

### Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Pipeline Execution Order

```bash
# 1. Generate Bronze data
python src/bet/pipelines/create_bronze_dataset.py

# 2. Generate Silver data
python src/bet/pipelines/create_silver_dataset.py

# 3. Generate Gold data
python src/bet/pipelines/create_gold_dataset.py

# 4. Train the model
python src/bet/models/logistic_regression.py

# 5. Run batch inference for a scoring date
python src/bet/models/inference.py 2024-06-20

# 6. Run backtesting (optional, after training)
python src/bet/evaluation/backtest.py
```

### Running with Docker

```bash
docker build -t bet-project .

docker run --rm -v "$(pwd):/app" bet-project bronze
docker run --rm -v "$(pwd):/app" bet-project silver
docker run --rm -v "$(pwd):/app" bet-project gold
docker run --rm -v "$(pwd):/app" bet-project train
docker run --rm -v "$(pwd):/app" bet-project backtest
docker run --rm -v "$(pwd):/app" bet-project inference 2024-06-20
```

## Model Training Details

- **Algorithm:** Logistic Regression (Spark ML)
- **Preprocessing:** StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler → LogisticRegression
- **Split strategy:** Chronological — Train 70% / Val 15% / Test 15%
- **Cross-validation:** 3-fold CV with AUPR as primary metric
- **Hyperparameter grid:** regParam ∈ [0.01, 0.1, 0.5], elasticNetParam ∈ [0.0, 0.5]
- **Class imbalance:** Handled via class weights (minority class upweighted)
- **Threshold optimization:** Tests 19 thresholds (5%-95%) on validation set

### MLflow Tracking

Each training run logs:
- Hyperparameters, evaluation metrics, threshold choices
- Feature importance coefficients
- Precision-recall curve
- Model artifacts with signature
- Git metadata (commit, branch) when available
- System info (Python version, platform)

### Important: Production Model Alias

After training, you must **manually assign the `production` alias** to the desired registered model version in MLflow. Both `inference.py` and `backtest.py` load the model using:

```
models:/SparkLogisticRegression_train@production
```

## Configuration

Key constants in `src/bet/utils/constants.py`:

| Constant | Value | Description |
|---|---|---|
| `CHURN_INACTIVITY_DAYS` | 7 | Days of inactivity defining churn |
| `MIN_HISTORY_DAYS` | 30 | Minimum data history for training |
| `RANDOM_SEED` | 42 | Reproducibility seed |
| `SPARK_DRIVER_MEMORY` | 12g | Spark driver memory |
| `MLFLOW_EXPERIMENT_NAME` | churn_prediction | MLflow experiment name |
| `LOW_RISK_THRESHOLD` | 0.25 | Probability below this = low risk |
| `MEDIUM_RISK_THRESHOLD` | 0.50 | Probability below this = medium risk |
| `HIGH_RISK_THRESHOLD` | 0.75 | Probability above this = high risk |

## Coding Conventions

- **Docstrings:** All functions have Google-style docstrings with Args, Returns, and Raises sections
- **Type hints:** Used throughout (Tuple, List, Dict, DataFrame, etc.)
- **Logging:** Uses `get_logger()` from `logging_utils` — no raw `print()` statements in production code
- **Modularity:** Pipeline stages are separated into focused functions with clear single responsibilities
- **DRY principle:** Common operations (reading tables, extracting player_idx) are centralized in `data_utils.py`
- **Constants over magic numbers:** All configurable values live in `constants.py` or `config.py`

## Package Name

The package is named **`bet`** (installed via `setup.py`). All imports use `from bet.utils...`, `from bet.ingestion...`, etc.
