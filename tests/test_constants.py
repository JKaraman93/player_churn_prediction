from bet.utils.constants import CHURN_INACTIVITY_DAYS, MIN_PLAYERS_FOR_TRAINING, ROLLING_WINDOW_7_DAYS, ROLLING_WINDOW_30_DAYS

def test_churn_inactivity_days():
    assert CHURN_INACTIVITY_DAYS == 7

def test_min_players_for_training():
    assert MIN_PLAYERS_FOR_TRAINING == 100


def test_rolling_window_constants():
    assert ROLLING_WINDOW_7_DAYS == 7
    assert ROLLING_WINDOW_30_DAYS == 30
