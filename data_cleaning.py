from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, trim, when
from pyspark.sql.types import StructType, StringType, DoubleType, TimestampType

spark = SparkSession.builder \
    .appName("KafkaStreamCleaning") \
    .getOrCreate()

schema = StructType() \
    .add("UserID", StringType()) \
    .add("Amount", DoubleType()) \
    .add("Location", StringType()) \
    .add("Time", TimestampType()) \
    .add("Device", StringType()) \
    .add("Merchant", StringType())

df = spark.readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", "kafka:9092") \
  .option("subscribe", "raw_data") \
  .option("startingOffsets", "latest") \
  .load()

json_df = df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

clean_df = (
    json_df
    .withColumn("Location", trim(col("Location")))
    .withColumn("Amount", when(col("Amount").isNull() | (col("Amount") < 0), 0)
                .otherwise(col("Amount").cast("double")))
    .filter(col("UserID").isNotNull() & col("Merchant").isNotNull())
)

clean_df.writeStream \
    .foreachBatch(lambda batch_df, batch_id:
        batch_df.write
            .format("jdbc")
            .option("url", "jdbc:mysql://mysql:3306/transactions")
            .option("dbtable", "clean_data")
            .option("user", "root")
            .option("password", "root")
            .option("driver", "com.mysql.cj.jdbc.Driver")
            .mode("append")
            .save()
    ) \
    .start() \
    .awaitTermination()
