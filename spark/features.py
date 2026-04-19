from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    avg,
    stddev,
    max as spark_max,
    when,
    lit
)

def main():

    print("Starting Spark Session...")

    spark = SparkSession.builder \
        .appName("JEE_Feature_Engineering_v2") \
        .getOrCreate()

    print("Reading cleaned data from HDFS...")

    df = spark.read.parquet("hdfs://namenode:9000/processed/clean")

    # -----------------------------------------
    # Create institute_type
    # -----------------------------------------
    df = df.withColumn(
        "institute_type",
        when(col("institute_short").contains("IIT"), lit("IIT"))
        .otherwise(lit("NIT"))
    )

    # -----------------------------------------
    # Ensure required columns exist
    # -----------------------------------------
    if "program_duration" not in df.columns:
        df = df.withColumn("program_duration", lit(4))

    if "degree_short" not in df.columns:
        df = df.withColumn("degree_short", lit("BTech"))

    # -----------------------------------------
    # Main grouping columns
    # -----------------------------------------
    group_cols = [
        "institute_type",
        "institute_short",
        "program_name",
        "category",
        "quota",
        "pool",
        "round_no",
        "program_duration",
        "degree_short",
        "year"
    ]

    print("Building rich feature table...")

    features = df.groupBy(*group_cols).agg(
        avg("opening_rank").alias("opening_rank"),
        avg("closing_rank").alias("closing_rank"),
        avg("closing_rank").alias("closing_rank_avg"),
        stddev("closing_rank").alias("closing_rank_std"),
        spark_max("year").alias("latest_recorded_year")
    )

    features = features.fillna({
        "closing_rank_std": 0.0
    })

    print("Writing upgraded features to HDFS...")

    features.write.mode("overwrite").parquet(
        "hdfs://namenode:9000/processed/features"
    )

    print("Done.")
    print("Saved: hdfs://namenode:9000/processed/features")

    spark.stop()


if __name__ == "__main__":
    main()