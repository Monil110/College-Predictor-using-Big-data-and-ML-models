from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.sql.functions import col, upper, trim, input_file_name, regexp_extract

def main():
    print("Starting Spark Session for NEET Data Ingestion...")
    spark = SparkSession.builder \
        .appName("NEET_Data_Ingest") \
        .getOrCreate()

    # Define strict schema handling DoubleType for fractional ranks (e.g., 1.01)
    schema = StructType([
        StructField("SNo", IntegerType(), True),
        StructField("Rank", DoubleType(), True),
        StructField("Allotted Quota", StringType(), True),
        StructField("Institute", StringType(), True),
        StructField("Course", StringType(), True),
        StructField("Allotted Category", StringType(), True),
        StructField("Candidate Category", StringType(), True),
        StructField("Remarks", StringType(), True)
    ])

    print("Reading CSV from local data folder...")
    df = spark.read.csv("/app/data/raw/neet", header=True, schema=schema)

    # Step 4: Data Cleaning in Spark
    print("Cleaning Data...")
    df_clean = df.dropna(subset=["Rank", "Institute"])
    
    # Ensure rank is valid
    df_clean = df_clean.filter(col("Rank") > 0)
    
    # Standardize category string
    df_clean = df_clean.withColumn("Allotted Category", upper(trim(col("Allotted Category"))))
    
    # Dynamically extract year from filename e.g. "hdfs://.../2020r2.csv"
    df_clean = df_clean.withColumn("year", regexp_extract(input_file_name(), r'(\d{4})', 1).cast("int"))
    
    # Validate the regex matched a valid year
    df_clean = df_clean.filter(col("year").isNotNull())

    # Rename to clean columns without spaces
    df_clean = df_clean.withColumnRenamed("SNo", "sno") \
                       .withColumnRenamed("Rank", "rank") \
                       .withColumnRenamed("Allotted Quota", "quota") \
                       .withColumnRenamed("Institute", "institute") \
                       .withColumnRenamed("Course", "course") \
                       .withColumnRenamed("Allotted Category", "category") \
                       .withColumnRenamed("Candidate Category", "candidate_category") \
                       .withColumnRenamed("Remarks", "remarks")

    # Save as partitioned Parquet
    print("Writing cleanly partitioned Parquet files to local data folder...")
    df_clean.write.mode("overwrite").partitionBy("year", "category").parquet("/app/data/processed/neet_cleaned")
    
    print("Ingestion Pipeline Complete! Parquet data saved to: /app/data/processed/neet_cleaned")
    spark.stop()

if __name__ == "__main__":
    main()
