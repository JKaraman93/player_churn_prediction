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
from bet.utils.constants import RANDOM_SEED, CHURN_INACTIVITY_DAYS, BRONZE_DATA_PATH


@dataclass
class DataGenConfig:
    """Configuration for synthetic data generation pipeline.
    
    Attributes:
        seed: Random seed for reproducible data generation
        num_players: Number of synthetic players to generate
        start_date: Start date for generated data (YYYY-MM-DD format)
        end_date: End date for generated data (YYYY-MM-DD format)
        active_lambda: Lambda parameter for active player distribution
        at_risk_lambda: Lambda parameter for at-risk player distribution
        churn_inactivity_days: Days of inactivity that defines a churned player
        bronze_path: Base directory path for bronze layer storage
    """
    seed: int = RANDOM_SEED
    num_players: int = 1000
    start_date: str = '2024-01-01'
    end_date: str = '2024-06-30'
    active_lambda: float = 1.2
    at_risk_lambda: float = 0.4
    churn_inactivity_days: int = CHURN_INACTIVITY_DAYS
    bronze_path: str = BRONZE_DATA_PATH
    