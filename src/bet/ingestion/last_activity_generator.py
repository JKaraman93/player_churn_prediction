
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

from pyspark.sql import functions as F

def generate_last_activity(events):
    first_last_session= (
        events
        .filter(F.col('event_type')=='session')
        .groupBy("player_idx")
        .agg(
            F.to_date(F.min("event_ts")).alias("first_session_date"),
            F.to_date(F.max("event_ts")).alias("last_session_date"),)
    )
    first_last_fin_tr= (
        events
        .filter(F.col('event_type')!='session')
        .groupBy("player_idx")
        .agg(
            F.to_date(F.min("event_ts")).alias("first_financial_date"),
            F.to_date(F.max("event_ts")).alias("last_financial_date"),)
    )
    first_last_event= (
        events
        .groupBy("player_idx")
        .agg(
            F.to_date(F.min("event_ts")).alias("first_event_date"),
            F.to_date(F.max("event_ts")).alias("last_event_date"),)
    )

    first_last_activity = (first_last_event
    .join(first_last_session, how='left', on='player_idx')
    .join(first_last_fin_tr, how='left', on='player_idx')
    )

    return first_last_activity
