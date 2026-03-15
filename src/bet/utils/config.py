"""
Configuration Module: Pipeline Parameters

Defines the DataGenConfig dataclass with all configurable parameters for
the data generation and training pipeline.

Parameters:
- seed: Random seed for reproducibility
- num_players: Number of players to generate
- start_date / end_date: Date range for data generation
- active_lambda / at_risk_lambda: Activity rate parameters
- churn_inactivity_days: Threshold for churn definition
- bronze_path: Base path for bronze layer storage
"""

from dataclasses import dataclass 

@dataclass
class DataGenConfig:
    seed: int = 42
    num_players: int = 1000
    start_date: str = '2024-01-01'
    end_date: str = '2024-06-30'

    active_lambda: float = 1.2
    at_risk_lambda: float = 0.4

    churn_inactivity_days: int = 7

    bronze_path: str = '/mnt/bronze'
    