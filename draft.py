
from src.utils.spark_session import get_spark
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = get_spark()



data = [
    ("A", "2024-01-01", 100),
    ("A", "2024-01-03", 200),
    ("A", "2024-01-05", 50),
    ("B", "2024-01-02", 300),
    ("B", "2024-01-04", 150),
]

df = spark.createDataFrame(
    data,
    ["customer_id", "date", "amount"]
).withColumn("date", F.to_date("date"))

df.show()



w = Window.partitionBy("customer_id").orderBy("date")

df.withColumn(
    "running_total",
    F.sum("amount").over(w)
).show()


df.withColumn(
    "prev_amount",
    F.lag("amount").over(w)
).withColumn(
    "delta",
    F.col("amount") - F.col("prev_amount")
).withColumn(
    "next_amount",
    F.lead("amount").over(w)
).withColumn(
    "delta2",
    F.col("amount") - F.col("next_amount")
).show()


w_latest = Window.partitionBy("customer_id").orderBy(F.col("date").desc())

df.withColumn(
    "rn",
    F.row_number().over(w_latest)


).filter(F.col("rn") == 1).show()


w_rank = Window.partitionBy("customer_id").orderBy(F.col("amount").desc())

df.withColumn(
    "rank",
    F.rank().over(w_rank)
).show()


w_rank = Window.partitionBy("customer_id").orderBy(F.col("amount").desc())

df.withColumn(
    "rank",
    F.row_number().over(w_rank)
).show()

w_roll = (
    Window
    .partitionBy("customer_id")
    .orderBy("date")
    .rowsBetween(-1, 0)
)

df.withColumn(
    "moving_avg",
    F.avg("amount").over(w_roll)
).show()



labels = (
    last_activity
    .withColumn(
        "churn_7d",
        F.expr(
            f"datediff('{config.end_date}', last_session_date) >= {config.churn_inactivity_days}"
        )
    )
    .withColumn("reference_date", F.lit(config.end_date))
)










#silver_transactions.filter(F.col('player_id')=='P36956').limit(6).show()
#6609558




