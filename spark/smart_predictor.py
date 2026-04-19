from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("Smart Predictor") \
    .config("spark.network.timeout", "600s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()

rank = int(input("Enter Rank: "))
cat = input("Enter Category: ").strip()

df = spark.read.parquet("hdfs://namenode:9000/processed/clean")

df = df.filter(
    (col("category") == cat) &
    (col("closing_rank") > 0) &
    (col("opening_rank") >= 0) &
    (col("opening_rank") <= col("closing_rank"))
).select(
    "institute_short","program_name","closing_rank"
).dropDuplicates().cache()

df.count()

safe = df.filter(col("closing_rank") >= int(rank * 1.3)).orderBy("closing_rank")
moderate = df.filter((col("closing_rank") >= rank) & (col("closing_rank") < int(rank * 1.3))).orderBy("closing_rank")
dream = df.filter((col("closing_rank") >= int(rank * 0.8)) & (col("closing_rank") < rank)).orderBy(col("closing_rank").desc())

print("\nSAFE COLLEGES")
safe.show(15, False)

print("\nMODERATE COLLEGES")
moderate.show(15, False)

print("\nDREAM COLLEGES")
dream.show(15, False)

spark.stop()