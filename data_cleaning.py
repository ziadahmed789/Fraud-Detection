from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType, DoubleType, TimestampType
from pyspark.sql.functions import trim, when

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
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "transactions") \
    .load()

json_df = df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

clean_df = (
    json_df
    .withColumn("Location", trim(col("Location"))) 
    .withColumn("Amount", when(col("Amount").isNull() | (col("Amount") < 0), 0).otherwise(col("Amount").cast("double")))  # يصلح القيم الغلط
    .filter(col("UserID").isNotNull() & col("Merchant").isNotNull()) 
)
query = clean_df.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("path", "hdfs://namenode:9000/data/cleaned/transactions/") \
    .option("checkpointLocation", "hdfs://namenode:9000/data/checkpoints/transactions/") \
    .start()

