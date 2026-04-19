from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import col, upper, trim

def main():
    print("Starting Spark Session for Data Ingestion...")
    spark = SparkSession.builder \
        .appName("JEE_Data_Ingest") \
        .getOrCreate()

    # Step 3: Define strict schema for robustness
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("year", IntegerType(), True),
        StructField("institute_type", StringType(), True),
        StructField("round_no", IntegerType(), True),
        StructField("quota", StringType(), True),
        StructField("pool", StringType(), True),
        StructField("institute_short", StringType(), True),
        StructField("program_name", StringType(), True),
        StructField("program_duration", StringType(), True),
        StructField("degree_short", StringType(), True),
        StructField("category", StringType(), True),
        StructField("opening_rank", IntegerType(), True),
        StructField("closing_rank", IntegerType(), True),
        StructField("is_preparatory", IntegerType(), True)
    ])

    print("Reading CSV from HDFS...")
    df = spark.read.csv("hdfs://namenode:9000/raw/jee.csv", header=True, schema=schema)

    # Step 4: Data Cleaning in Spark
    print("Cleaning Data...")
    df_clean = df.dropna(subset=["opening_rank", "closing_rank"])
    
    # Drop preparatory courses to avoid skewed predictive models
    df_clean = df_clean.filter(col("is_preparatory") == 0)
    
    # Standardize categories
    df_clean = df_clean.withColumn("category", upper(trim(col("category"))))
    
    # Data invariant validations
    df_clean = df_clean.filter(col("opening_rank") > 0)
    df_clean = df_clean.filter(col("opening_rank") <= col("closing_rank"))
    
    # Save as partitioned Parquet to HDFS
    print("Writing cleanly partitioned Parquet files to HDFS...")
    df_clean.write.mode("overwrite").partitionBy("year", "category").parquet("hdfs://namenode:9000/processed/clean")
    
    print("Ingestion Pipeline Complete! Parquet data saved to HDFS: hdfs://namenode:9000/processed/clean")
    spark.stop()

if __name__ == "__main__":
    main()