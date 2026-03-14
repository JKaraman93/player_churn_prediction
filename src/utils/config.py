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
    