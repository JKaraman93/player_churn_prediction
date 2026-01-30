from pyspark.sql import functions as F
from pyspark.sql.types import StringType
import random

def generate_player_profiles(spark, config):
    spark.sparkContext.setCheckpointDir("/tmp/checkpoints")
    random.seed(config.seed)

    df = spark.range(0, config.num_players).withColumnRenamed("id", "player_idx")

    ## player id ##
    df = df.withColumn("player_id", F.concat(F.lit("P"), F.col("player_idx")))
    
    ## registration date ##
    df = df.withColumn(
                "registration_date",
                F.expr(f"date_add('{config.start_date}', cast(rand({config.seed}) * 120 as int))")
    )
    ## Country ##
    df = df.withColumn(
                "country",
                F.expr(f"""
                    CASE
                    WHEN rand() < 0.4 THEN 'GR'
                    WHEN rand() < 0.6 THEN 'EN'
                    WHEN rand() < 0.8 THEN 'DE'
                    ELSE 'OTHER'
                    END
                    """) )

    ## age bucket ## 
    df = df.withColumn('age_bucket',
                    F.expr(f"""
                        CASE 
                            WHEN rand() < 0.2 THEN "18-24"
                            WHEN rand() < 0.4 THEN "25-34"
                            WHEN rand() < 0.6 THEN "35-44"
                            ELSE "45+"
                        END
                            """))

    ## device type ##
    df = df.withColumn('device_type',
                    F.expr(f"""
                    CASE 
                        WHEN rand() < 0.6 THEN "desktop"
                        ELSE  "mobile"
                    END
                            """))


    ## acquicition channel ## 
    df = df.withColumn("acquisition_channel",
                    F.expr(f"""
                            CASE 
                            WHEN rand() < 0.3 THEN "organic"
                            WHEN rand() < 0.6 THEN "paid"
                            ELSE "affiliate"
                            END
                            """))

    df.show()
    return df





