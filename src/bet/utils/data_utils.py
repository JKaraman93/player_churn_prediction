"""
Utilities Module: Common Data Transformation Helpers

Provides reusable functions for common data operations to promote
DRY (Don't Repeat Yourself) principles and reduce code duplication.
"""

from typing import List
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import LongType


def extract_player_idx_from_id(df: DataFrame, id_column: str = 'player_id', 
                               new_column: str = 'player_idx') -> DataFrame:
    """
    Extract numeric player index from player_id string.
    
    Removes all non-numeric characters and casts to long integer.
    Drops the original player_id column after extraction.
    
    Args:
        df: Input Spark DataFrame
        id_column: Name of the column containing player_id strings (default: 'player_id')
        new_column: Name for the new player_idx column (default: 'player_idx')
        
    Returns:
        DataFrame with player_idx column added and original id_column dropped
    """
    return (df
            .withColumn(new_column, 
                       F.regexp_replace(id_column, "[^0-9]", "").cast(LongType()))
            .drop(id_column))


def add_player_idx_to_tables(tables: dict) -> dict:
    """
    Apply player ID extraction to multiple tables.
    
    Convenience function for processing multiple DataFrames that contain
    player_id columns from Silver layer data.
    
    Args:
        tables: Dictionary with table names as keys and DataFrames as values
        
    Returns:
        Dictionary with same keys and transformed DataFrames as values
    """
    return {name: extract_player_idx_from_id(df) for name, df in tables.items()}


def read_silver_tables(spark) -> dict:
    """
    Read all Silver layer tables into memory.
    
    Provides convenient single-point access for loading standard Silver layer
    tables: players, sessions, transactions, and money_events.
    
    Args:
        spark: Spark session instance
        
    Returns:
        Dictionary with table names as keys and DataFrames as values
    """
    from bet.utils.constants import SILVER_DATA_PATH
    
    return {
        'players': spark.read.parquet(f"{SILVER_DATA_PATH}/players"),
        'sessions': spark.read.parquet(f"{SILVER_DATA_PATH}/sessions"),
        'transactions': spark.read.parquet(f"{SILVER_DATA_PATH}/transactions"),
        'money_events': spark.read.parquet(f"{SILVER_DATA_PATH}/money_events"),
    }


def read_gold_tables(spark) -> dict:
    """
    Read all Gold layer tables into memory.
    
    Provides convenient single-point access for loading standard Gold layer
    tables: player_behavior, player_snapshot, and labels.
    
    Args:
        spark: Spark session instance
        
    Returns:
        Dictionary with table names as keys and DataFrames as values
    """
    from bet.utils.constants import GOLD_DATA_PATH
    
    return {
        'player_behavior': spark.read.parquet(f"{GOLD_DATA_PATH}/player_behavior"),
        'player_snapshot': spark.read.parquet(f"{GOLD_DATA_PATH}/player_snapshot"),
        'labels': spark.read.parquet(f"{GOLD_DATA_PATH}/labels"),
    }
