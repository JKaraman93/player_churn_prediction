# Player Churn Prediction – End‑to‑End ML System

## 1. Project Overview

This project implements a **production‑oriented machine learning pipeline** to predict whether a gaming player will **complete a 7‑day inactivity (churn) period within the next 7 days**.

Unlike simple player‑level churn models, the prediction unit here is:

> **(player, reference_date)** → probability of churn completion in *(t, t+7]*

This enables **daily risk scoring**, early intervention, and realistic deployment in gaming or subscription environments.

Model targets players with at **least 30 days historical activity** to ensure stable rolling feature computation and prevent cold-start bias.


---

## 2. Business Objective

The system produces a **daily churn‑risk table** that can be consumed by CRM or retention systems:

| player_idx | score_date | p_churn | risk_segment |
| ---------- | ---------- | ------- | ------------ |

This supports:

* Early retention campaigns
* Dynamic bonus allocation
* Monitoring of player health over time

---

## 3. Data Architecture

The project follows a **Bronze → Silver → Gold** medallion structure.

### Bronze

Raw synthetic data:

* Players
* Sessions
* Financial transactions

### Silver

Cleaned, time‑consistent event tables used as **source of truth** for ML features.

### Gold

ML‑ready datasets:

#### Player Snapshot

Static attributes per player:

* Country, age bucket, acquisition channel
* Registration date
* Lifecycle / risk segment
* Current balance
* Last session date, inactivity days

#### Player Behavior (rolling features)

Rolling **7‑day and 30‑day** aggregates per *(player, reference_date)*:

* Net financial and game results
* Session counts and duration
* Bet behavior
* Withdrawal statistics and ratios
* Historical balances (7d / 30d ago)

#### Labels

Binary target:

> **next_7d_churn = 1** if a 7‑day inactivity window will complete in the next 7 days.

This formulation approximates a **discrete survival / hazard prediction** problem.

---

## 4. Feature Engineering Principles

Key design rules applied:

* **Strict time causality** → features use only data ≤ reference_date
* **Rolling windows** → 7‑day and 30‑day behavioral summaries
* **Zero‑activity preservation** → inactive players retained with zero values
* **Training–serving consistency** → identical window definitions offline and in production

---

## 5. Modeling Approach

### Algorithm

Baseline model:

* **Logistic Regression (Spark ML)**

Chosen for:

* Interpretability
* Fast training on large data
* Strong baseline for tabular churn problems

### Preprocessing Pipeline

Spark ML pipeline includes:

* StringIndexing + One‑Hot Encoding for categorical variables
* Standard scaling for numeric features
* Vector assembly
* Class weighting for imbalance handling

### Validation Strategy

* **Time‑based train / validation / test split**
* **Cross‑validation** on training period
* **Area Under Precision‑Recall (AUPR)** as primary metric

Why AUPR:

> Churn is **imbalanced**, so PR is more informative than ROC.

### Threshold Optimization

Churn probability is converted to **business risk segments** by selecting a threshold that balances:

* Precision (avoid unnecessary incentives)
* Recall (capture real churners)
* F1 score (overall trade‑off)

---

## 6. Experiment Tracking (MLflow)

The training workflow logs:

* Hyperparameters
* Metrics (AUPR, precision, recall, F1)
* Selected decision threshold
* Spark ML pipeline model artifact
* Environment metadata (Python version, platform, Git commit)
* Dataset fingerprints / checksums

This ensures **full reproducibility and auditability**.

---

## 7. Evaluation Interpretation

Because the unit is *(player, day)* rather than player‑level:

* Metrics are **per‑day predictions**
* Multiple positive labels may correspond to the **same churn episode**

Therefore:

> Precision ≠ % of churned players correctly predicted

Instead, metrics measure **daily early‑warning quality**.

---

## 8. Production Inference Design

### Core Principle

Inference must:

> Recompute rolling features from **Silver source data** using only data ≤ score_date.

Gold historical aggregates are **not trusted** in production to avoid leakage or staleness.

### Daily Batch Scoring

For each day *t‑1*:

1. Load Silver sessions, transactions, and player attributes
2. Aggregate events within **[t‑29, t]**
3. Build feature vector using the **MLflow‑stored pipeline**
4. Predict churn probability
5. Map probability → **risk segment**
6. Write **append‑only prediction table**

This creates a **deployable ML data product**.

---

## 9. Current Status

Completed:

* Synthetic data generation
* Gold feature engineering with rolling windows
* Leakage‑free churn label construction
* Logistic regression pipeline with class weighting
* Cross‑validation and threshold tuning
* MLflow experiment tracking and model logging
* Production‑aligned inference design

---

## 10. Next Steps

Planned improvements toward full MLOps maturity:

* **Daily batch inference script** with backfill support
* Prediction table monitoring (drift, score distribution, alerting)
* Advanced models (Gradient Boosting, XGBoost, LightGBM)
* Survival analysis / time‑to‑churn modeling
* CI/CD automation and scheduled retraining
* Deployment on cloud Spark / Databricks / Kubernetes

---

## 11. Tech Stack

* Python
* PySpark
* Spark MLlib
* MLflow
* Parquet data lake structure

---
