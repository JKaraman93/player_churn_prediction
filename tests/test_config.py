from bet.utils.config import DataGenConfig
from bet.utils.constants import RANDOM_SEED, CHURN_INACTIVITY_DAYS, BRONZE_DATA_PATH

def test_data_gen_config_defaults():
    d = DataGenConfig()
    assert d.seed==RANDOM_SEED 
    assert d.churn_inactivity_days == CHURN_INACTIVITY_DAYS
    assert d.bronze_path == BRONZE_DATA_PATH