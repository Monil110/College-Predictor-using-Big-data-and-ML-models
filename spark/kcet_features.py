from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, stddev, max as spark_max

def main():
    print("Starting Spark Session for KCET Feature Engineering...")
    spark = SparkSession.builder \
        .appName("KCET_Feature_Engineering") \
        .getOrCreate()

    print("Reading cleaned data from local data folder...")
    df = spark.read.parquet("/app/data/processed/kcet_cleaned")

    # Main grouping columns
    group_cols = [
        "college_name",
        "course_name",
        "category",
        "base_category",
        "quota",
        "region",
        "year"
    ]

    print("Building KCET feature table aggregations...")
    # Calculating the maximum rank acquired per group represents the CLOSING RANK
    features = df.groupBy(*group_cols).agg(
        spark_max("rank").alias("closing_rank"),
        avg("rank").alias("closing_rank_avg"),
        stddev("rank").alias("closing_rank_std"),
        spark_max("year").alias("latest_recorded_year")
    )

    # Fill invalid variance
    features = features.fillna({"closing_rank_std": 0.0})

    print("Writing KCET upgraded features to local data folder...")
    features.write.mode("overwrite").parquet("/app/data/processed/kcet_features")

    print("Done.")
    print("Saved: /app/data/processed/kcet_features")

    spark.stop()

if __name__ == "__main__":
    main()
