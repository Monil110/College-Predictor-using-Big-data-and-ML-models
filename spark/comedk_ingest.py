import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, upper, trim, lit
from pyspark.sql.types import DoubleType


def main():
    print("Starting Spark Session for COMEDK Data Ingestion...")
    spark = SparkSession.builder \
        .appName("COMEDK_Data_Ingest") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    raw_dir = "hdfs://namenode:9000/comedk/raw"
    out_path = "hdfs://namenode:9000/comedk/processed/cleaned"

    final_df = None

    for year in [2023, 2024, 2025]:
        path = f"{raw_dir}/comedk{year}.csv"
        print(f"\nReading {path}...")

        try:
            df = spark.read \
                .option("header", True) \
                .option("multiLine", True) \
                .option("escape", '"') \
                .option("quote", '"') \
                .option("inferSchema", False) \
                .csv(path)
        except Exception as e:
            print(f"[SKIP] Failed to read {path}: {e}")
            continue

        print(f"  Rows: {df.count()}")

        # ✅ Direct mapping (NO unpivot needed)
        df_clean = df \
            .withColumn("college_code", trim(col("College Code"))) \
            .withColumn("college_name", trim(col("College Name"))) \
            .withColumn("category", upper(trim(col("Seat Category")))) \
            .withColumn("course_name", trim(col("Branch Name"))) \
            .withColumn("rank", col("Cutoff Rank").cast(DoubleType())) \
            .withColumn("year", lit(year)) \
            .select("college_code", "college_name", "category", "course_name", "rank", "year")

        # Remove bad rows
        df_clean = df_clean.filter(
            col("college_code").isNotNull() &
            col("college_name").isNotNull() &
            col("course_name").isNotNull() &
            col("rank").isNotNull() &
            (col("rank") > 0)
        )

        row_count = df_clean.count()
        print(f"  Year {year}: {row_count} valid rows")

        if row_count == 0:
            continue

        if final_df is None:
            final_df = df_clean
        else:
            final_df = final_df.unionByName(df_clean)

    if final_df is None:
        print("[ERROR] No data processed")
        spark.stop()
        return

    # Keep only needed categories
    final_df = final_df.filter(col("category").isin("GM", "KKR"))

    # Remove duplicates
    final_df = final_df.dropDuplicates(
        ["college_code", "college_name", "category", "course_name", "year"]
    )

    total_rows = final_df.count()
    print(f"\nTotal rows after cleaning: {total_rows}")

    print(f"\nWriting cleaned data to: {out_path}")
    final_df.write \
        .mode("overwrite") \
        .parquet(out_path)

    print("COMEDK Ingestion Complete!")
    spark.stop()


if __name__ == "__main__":
    main()