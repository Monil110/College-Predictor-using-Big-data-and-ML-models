from pyspark.sql import SparkSession

def main():
    print("Starting Spark Session for Hive Metastore Registration...")
    # Enabling Hive support
    spark = SparkSession.builder \
        .appName("JEE_Hive_Setup") \
        .config("spark.sql.catalogImplementation", "hive") \
        .enableHiveSupport() \
        .getOrCreate()
    
    print("Setting up jee_db...")
    spark.sql("CREATE DATABASE IF NOT EXISTS jee_db")
    spark.sql("USE jee_db")
    
    # Step 6: Set up Hive metastore pointing to parquet partitions 
    print("Registering clean_data table...")
    spark.sql("""
        CREATE TABLE IF NOT EXISTS clean_data
        USING parquet
        OPTIONS (path 'hdfs://localhost:9000/processed/clean')
    """)
    
    print("Registering features table...")
    spark.sql("""
        CREATE TABLE IF NOT EXISTS features
        USING parquet
        OPTIONS (path 'hdfs://localhost:9000/processed/features')
    """)
    
    print("Metastore setup complete. You can now use Spark SQL to query 'jee_db.clean_data' and 'jee_db.features'.")
    spark.stop()

if __name__ == "__main__":
    main()
