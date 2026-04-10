"""
Spark Session Factory: Spark Configuration and Initialization

Provides the get_spark() function to create and return configured
SparkSession instances for use across the project.

Configuration:
- Local mode with all CPU cores
- 12GB driver and executor memory
- Optimized for both single-machine development and small clusters
"""

from pyspark.sql import SparkSession
from bet.utils.constants import SPARK_DRIVER_MEMORY, SPARK_EXECUTOR_MEMORY


def get_spark(app_name: str = 'BetAnalytics') -> SparkSession:
    """
    Create and return a configured SparkSession instance.
    
    Sets up Spark with optimized memory configuration for data processing
    on a single machine or small cluster.
    
    Args:
        app_name: Name for the Spark application (default: 'BetAnalytics')
        
    Returns:
        Configured SparkSession instance ready for use
    """
    return (SparkSession.builder
            .master("local[*]")
            .appName(app_name)
            .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
            .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
            .getOrCreate())


