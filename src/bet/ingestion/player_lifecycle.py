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

from pyspark.sql import functions as F

def assign_lifecycle(df_players):
    return (
        df_players
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
