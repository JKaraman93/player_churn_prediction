from pyspark.sql import SparkSession

def get_spark(app_name :str= 'SyntheticDataGenerator'):
    return ( SparkSession.builder.master("local[*]").appName(app_name).getOrCreate())
