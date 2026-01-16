from pyspark.sql import SparkSession

def get_spark(app_name :str= 'SyntheticDataGenerator'):
    return ( SparkSession.builder.master("local[*]").appName(app_name).config("spark.driver.memory", "12g")  # driver memory
    .config("spark.executor.memory", "12g").getOrCreate())


