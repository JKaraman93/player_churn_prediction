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

def get_spark(app_name :str= 'SyntheticDataGenerator'):
    return ( SparkSession.builder.master("local[*]")
            .appName(app_name)
            .config("spark.driver.memory", "12g")  # driver memory
            .config("spark.executor.memory", "12g")
            .getOrCreate())


