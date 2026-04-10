"""
Player Profile Generation: Synthetic Player Data Creation

Generates synthetic player profiles with realistic demographics, account information,
and behavioral attributes. Creates the foundation for player-level features.

Generated attributes:
- player_idx: Unique player identifier
- Demographics: country, age_bucket, acquisition_channel, device_type
- Account: registration_date, current_balance
- Behavioral: lifecycle_stage (placeholder for downstream assignments)

Integration: Output is enhanced by lifecycle and risk segment assignment modules.
"""

from typing import Optional
from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.types import StringType
from bet.utils.config import DataGenConfig
from bet.utils.logging_utils import get_logger
import random

logger = get_logger(__name__)


def generate_player_profiles(spark: SparkSession, config: DataGenConfig) -> DataFrame:
    """
    Generate synthetic player profiles with demographics and account information.
    
    Creates a Spark DataFrame with realistic player attributes including:
    - Basic ID and dates
    - Geographic and demographic information
    - Acquisition and device information
    
    Uses random distribution for natural-looking data with configurable seed
    for reproducibility.
    
    Args:
        spark: Spark session instance
        config: DataGenConfig with seed and num_players parameters
        
    Returns:
        DataFrame with synthetic player profiles
    """
    logger.info(f"Generating profiles for {config.num_players} players")
    
    spark.sparkContext.setCheckpointDir("/tmp/checkpoints")
    random.seed(config.seed)

    df = spark.range(0, config.num_players).withColumnRenamed("id", "player_idx")

    # Generate player ID
    df = df.withColumn("player_id", F.concat(F.lit("P"), F.col("player_idx")))
    
    # Generate registration date
    df = df.withColumn(
        "registration_date",
        F.expr(f"date_add('{config.start_date}', cast(rand({config.seed}) * 120 as int))")
    )
    
    # Country distribution: 40% GR, 20% EN, 20% DE, 20% OTHER
    df = df.withColumn(
        "country",
        F.expr("""
            CASE
                WHEN rand() < 0.4 THEN 'GR'
                WHEN rand() < 0.6 THEN 'EN'
                WHEN rand() < 0.8 THEN 'DE'
                ELSE 'OTHER'
            END
        """)
    )

    # Age distribution: 20% each bucket
    df = df.withColumn(
        'age_bucket',
        F.expr("""
            CASE 
                WHEN rand() < 0.2 THEN '18-24'\n                WHEN rand() < 0.4 THEN '25-34'\n                WHEN rand() < 0.6 THEN '35-44'\n                ELSE '45+'\n            END
        """)
    )

    # Device type: 60% desktop, 40% mobile
    df = df.withColumn(
        'device_type',
        F.expr("""
            CASE 
                WHEN rand() < 0.6 THEN 'desktop'
                ELSE 'mobile'
            END
        """)
    )

    # Acquisition channel: 30% organic, 30% paid, 40% affiliate
    df = df.withColumn(
        "acquisition_channel",
        F.expr("""
            CASE 
                WHEN rand() < 0.3 THEN 'organic'
                WHEN rand() < 0.6 THEN 'paid'
                ELSE 'affiliate'
            END
        """)
    )

    logger.info(f"Generated player profiles:\n{df.limit(5).show(truncate=False)}\"")
    
    return df





