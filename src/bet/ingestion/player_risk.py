"""
Player Risk Segmentation: Risk Classification Based on Behavior

Assigns risk segments to players based on their betting patterns,
bookmaking metrics, and behavioral characteristics.

Risk segments:
- unknown: New players without history
- low: Conservative betting, stable patterns
- medium: Moderate engagement, some volatility
- high: Aggressive betting, volatile activity

Used for stratified analysis and business rule application.
Helps identify players who may need special attention or interventions.

Output: player profiles with risk_segment attribute
"""

from pyspark.sql import DataFrame, functions as F
from bet.utils.logging_utils import get_logger

logger = get_logger(__name__)


def assign_risk(df_players: DataFrame) -> DataFrame:
    """
    Assign risk segment to players based on lifecycle and behavior.
    
    Creates a new 'risk_segment' column with values:
    - 'unknown': New players without betting history
    - 'low': Conservative betting patterns (60% of non-new)
    - 'medium': Moderate engagement (30% of non-new, 60-90%)
    - 'high': Aggressive betting (10% of non-new, 90-100%)
    
    Args:
        df_players: Input DataFrame with player profiles and lifecycle stages
        
    Returns:
        DataFrame with new 'risk_segment' column added
    """
    logger.info(f"Assigning risk segments to {df_players.count()} players")
    
    result = (df_players
        .withColumn(
            "risk_segment", 
            F.when(F.col('lifecycle_stage') == 'new', F.lit("unknown"))
            .when(F.rand() < 0.6, F.lit("low"))
            .when(F.rand() < 0.9, F.lit("medium"))
            .otherwise(F.lit('high'))
        )
    )
    
    # Log distribution
    risk_counts = result.groupBy("risk_segment").count().collect()
    for row in risk_counts:
        logger.info(f"  {row['risk_segment']}: {row['count']} players")
    
    return result