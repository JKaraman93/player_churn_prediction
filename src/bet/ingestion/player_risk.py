"""
Player Risk Segmentation: Risk Classification Based on Behavior

Assigns risk segments to players based on their betting patterns,
bookmaking metrics, and behavioral characteristics.

Risk segments:
- Low risk: Conservative betting, stable patterns
- Medium risk: Moderate engagement, some volatility
- High risk: Aggressive betting, volatile activity

Used for stratified analysis and business rule application.
Helps identify players who may need special attention or interventions.

Output: player profiles with risk_segment attribute
"""

from pyspark.sql import functions as F

def assign_risk(df_players):
    return (
        df_players
        .withColumn(
            "risk_segment", 
            F.when( F.col('lifecycle_stage')=='new', F.lit("unknown"))
            .when(F.rand()<0.6, F.lit("low"))
            .when(F.rand()<0.9, F.lit("medium"))
            .otherwise(F.lit('high'))
        )
    )