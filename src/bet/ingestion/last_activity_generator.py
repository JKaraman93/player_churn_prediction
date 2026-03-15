"""
Last Activity Tracking: Recent Engagement Metrics

Computes last activity dates and inactivity streaks from session and transaction data.
Provides temporal indicators of player engagement for feature engineering.

Metrics computed:
- last_session_date: Most recent gaming session
- last_transaction_date: Most recent financial activity
- days_since_registration: Tenure in days
- current_inactivity_days: Days since last activity

Used in feature engineering to capture recency patterns and identify
players at risk of upcoming churn.
"""

from pyspark.sql import DataFrame, functions as F
from bet.utils.logging_utils import get_logger

logger = get_logger(__name__)


def generate_last_activity(events: DataFrame) -> DataFrame:
    """
    Compute first and last activity dates from events data.
    
    Extracts temporal boundaries for three event types:
    1. Session events (gaming activity)
    2. Financial events (transactions)
    3. Overall events (any activity)
    
    Creates aggregated dataset with columns:
    - first_session_date / last_session_date
    - first_financial_date / last_financial_date
    - first_event_date / last_event_date
    
    Args:
        events: DataFrame containing events with 'player_idx', 'event_type', and 'event_ts' columns
        
    Returns:
        DataFrame with first/last activity dates per player
    """
    logger.info(f"Computing last activity from {events.count()} events")
    
    # Session activities
    first_last_session = (events
        .filter(F.col('event_type') == 'session')
        .groupBy("player_idx")
        .agg(
            F.to_date(F.min("event_ts")).alias("first_session_date"),
            F.to_date(F.max("event_ts")).alias("last_session_date")
        )
    )
    
    # Financial activities
    first_last_fin_tr = (events
        .filter(F.col('event_type') != 'session')
        .groupBy("player_idx")
        .agg(
            F.to_date(F.min("event_ts")).alias("first_financial_date"),
            F.to_date(F.max("event_ts")).alias("last_financial_date")
        )
    )
    
    # All events
    first_last_event = (events
        .groupBy("player_idx")
        .agg(
            F.to_date(F.min("event_ts")).alias("first_event_date"),
            F.to_date(F.max("event_ts")).alias("last_event_date")
        )
    )

    # Join all results
    first_last_activity = (first_last_event
        .join(first_last_session, how='left', on='player_idx')
        .join(first_last_fin_tr, how='left', on='player_idx')
    )
    
    logger.info(f"Computed activity dates for {first_last_activity.count()} players")
    return first_last_activity
