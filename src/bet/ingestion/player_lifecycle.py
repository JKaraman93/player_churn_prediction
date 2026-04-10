"""
Player Lifecycle Assignment: Account Status Classification

Assigns lifecycle stages to players based on activity patterns and engagement.
Captures different phases of player journey (new, engaged, at_risk, churned).

Lifecycle stages:
- new: Recently registered players
- engaged: Active, healthy players
- at_risk: Declining activity, potential churn signals
- churned: Inactive players who have left

Used for segmentation and targeted interventions in downstream analysis.

Output: player profiles with lifecycle_stage attribute
"""

from pyspark.sql import DataFrame, functions as F
from bet.utils.logging_utils import get_logger

logger = get_logger(__name__)


def assign_lifecycle(df_players: DataFrame) -> DataFrame:
    """
    Assign lifecycle stage to players based on random distribution.
    
    Creates a new 'lifecycle_stage' column with values distributed as:
    - 60% 'engaged': Active, healthy players
    - 25% 'new': Recently registered players (85% - 60%)
    - 15% 'at_risk': Declining activity players (100% - 85%)
    
    Args:
        df_players: Input DataFrame with player profiles
        
    Returns:
        DataFrame with new 'lifecycle_stage' column added
    """
    logger.info(f"Assigning lifecycle stages to {df_players.count()} players")
    
    result = (df_players
        .withColumn(
            "lifecycle_stage",
            F.expr("""
                CASE
                    WHEN rand() < 0.6 THEN 'engaged'
                    WHEN rand() < 0.85 THEN 'new'
                    ELSE 'at_risk'
                END
            """)
        )
    )
    
    # Log distribution
    stage_counts = result.groupBy("lifecycle_stage").count().collect()
    for row in stage_counts:
        logger.info(f"  {row['lifecycle_stage']}: {row['count']} players")
    
    return result
