from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.sql.functions import col, upper, trim

def main():
    print("Starting Spark Session for KCET Data Ingestion...")
    spark = SparkSession.builder \
        .appName("KCET_Data_Ingest") \
        .getOrCreate()

    # Define schema according to target format
    schema = StructType([
        StructField("CollegeID", StringType(), True),
        StructField("CollegeName", StringType(), True),
        StructField("CourseName", StringType(), True),
        StructField("Category", StringType(), True),
        StructField("Cutoff_Rank", DoubleType(), True),
        StructField("Year", IntegerType(), True),
        StructField("Base_Category", StringType(), True),
        StructField("Quota", StringType(), True),
        StructField("Region", StringType(), True)
    ])

    print("Reading KCET CSV from local data folder...")
    df = spark.read.csv("/app/data/raw/kcet.csv", header=True, schema=schema)

    print("Cleaning Data...")
    df_clean = df.dropna(subset=["Cutoff_Rank", "CollegeName", "CourseName"])
    
    # Ensure rank is valid
    df_clean = df_clean.filter(col("Cutoff_Rank") > 0)
    
    # Text cleaning
    df_clean = df_clean.withColumn("Category", upper(trim(col("Category"))))
    df_clean = df_clean.withColumn("Base_Category", upper(trim(col("Base_Category"))))

    # Rename columns to snake_case matching implementation plan constraints
    df_clean = df_clean.withColumnRenamed("CollegeName", "college_name") \
                       .withColumnRenamed("CourseName", "course_name") \
                       .withColumnRenamed("Category", "category") \
                       .withColumnRenamed("Cutoff_Rank", "rank") \
                       .withColumnRenamed("Year", "year") \
                       .withColumnRenamed("Base_Category", "base_category") \
                       .withColumnRenamed("Quota", "quota") \
                       .withColumnRenamed("Region", "region") \
                       .withColumnRenamed("CollegeID", "college_id")

    print("Writing cleanly partitioned Parquet files to local data folder (HDFS pipeline via mount)...")
    # partitioned by year and category
    df_clean.write.mode("overwrite").partitionBy("year", "category").parquet("/app/data/processed/kcet_cleaned")
    
    print("KCET Ingestion Pipeline Complete! Parquet data saved to: /app/data/processed/kcet_cleaned")
    spark.stop()

if __name__ == "__main__":
    main()
