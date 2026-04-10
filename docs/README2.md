# Player Behavior Modeling – End‑to‑End ML Mini Project

## Overview

The project demonstrates the full ML lifecycle:

* Data generation and analysis at scale
* Feature engineering using distributed computing (Spark)
* Predictive modeling for player behavior
* MLOps practices including experiment tracking and deployment‑ready pipelines

The dataset is **synthetically generated**, but the modeling choices, data structures, and engineering patterns closely resemble real‑world production systems.

---

## Business Context

Online gaming platforms rely heavily on:

* Player engagement monitoring
* Churn prediction
* Personalized recommendations
* Dynamic content delivery

This project focuses on **player session behavior**, modeling how engagement evolves over time and how at‑risk players can be identified based on inactivity patterns.

---

## Project Objectives

1. Generate realistic, large‑scale player activity data using Spark
2. Model daily player behavior using probabilistic processes
3. Engineer time‑aware features suitable for ML models
4. Train and evaluate churn / engagement prediction models
5. Track experiments and metrics using MLflow
6. Structure the codebase for production‑readiness and scalability

---

## Architecture Overview

**Tech Stack**

* Python
* PySpark
* Spark SQL
* MLflow
* scikit‑learn
* Databricks‑compatible design

**Data Layers**

* **Bronze**: Raw synthetic events (sessions)
* **Silver**: Aggregated player‑day features
* **Gold**: Model‑ready feature tables

---

## Data Generation Strategy

### Synthetic Data Rationale

Due to the absence of real player data, a synthetic dataset is used. The goal is **behavioral realism**, not randomness.

Key principles:

* Explicit modeling of inactivity
* Stochastic session generation
* Lifecycle‑aware behavior changes

### Player‑Day Modeling

The core modeling unit is **player‑day**, created by cross‑joining:

* Player profiles
* A complete calendar date range

This ensures that days with **zero activity** are explicitly represented.

### Session Generation

Daily session counts are sampled using a **Poisson‑like process**, with different average rates (λ) depending on the player lifecycle stage:

* Engaged players → higher λ
* At‑risk players → lower λ
* Churned players → λ = 0

This allows engagement decay and churn patterns to **emerge naturally** from the data.

---

## Feature Engineering

Features are engineered at the **player‑day** level and include:

* Rolling session counts (7d / 14d / 30d)
* Days since last activity
* Activity frequency
* Engagement trend indicators

All features are computed using Spark window functions to ensure scalability.

---

## Modeling

### Problem Framing

Primary modeling tasks:

* Binary classification: churn vs active
* Risk scoring: probability of churn in the next N days

### Algorithms

* Logistic Regression (baseline)
* Gradient Boosted Trees

Models are trained using scikit‑learn on Spark‑generated feature tables.

---

## MLOps & Experiment Tracking

MLflow is used for:

* Experiment tracking
* Metric logging
* Model versioning

The project is structured to support:

* CI/CD pipelines
* Model retraining
* Future online inference integration

---

## Repository Structure

```
.
├── data_generation/
│   ├── config.py
│   ├── generate_players.py
│   ├── generate_sessions.py
│
├── features/
│   ├── player_day_features.py
│
├── models/
│   ├── train_model.py
│   ├── evaluate_model.py
│
├── notebooks/
│   ├── exploration.ipynb
│
├── requirements.txt
├── README.md
```

---

## How to Run

1. Set up the environment:

   ```bash
   pip install -r requirements.txt
   ```

2. Generate synthetic data using Spark

3. Build feature tables

4. Train and evaluate models

5. Review experiments in MLflow

---

## Design Decisions (Key Highlights)

* **Player‑day modeling** instead of event‑only modeling
* Explicit handling of inactivity
* Avoidance of Python UDFs where possible
* Deterministic randomness for reproducibility
* Databricks‑friendly architecture
