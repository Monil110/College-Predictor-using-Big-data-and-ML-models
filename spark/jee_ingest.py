"""
jee_ingest.py
─────────────────────────────────────────────────────────────────────────────
Stage 1: Read raw JEE CSV → clean → write partitioned Parquet.

No Hive metastore required.  Output lands in:
  /app/data/processed/jee_cleaned   (partitioned by year / institute_type / category)

Run:
    docker exec spark-master \
        /opt/spark/bin/spark-submit \
        --master spark://spark-master:7077 \
        /app/jee_ingest.py
─────────────────────────────────────────────────────────────────────────────
"""

import re
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

RAW_CSV_PATH  = "/app/data/raw/jee.csv"
CLEAN_PARQUET = "/app/data/processed/jee_cleaned"

# Keep only the final allocation round per year
FINAL_ROUNDS  = [6, 7]

# ─── Markdown-link stripper UDF ───────────────────────────────────────────────

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")

def _strip_md(text):
    """'[B.Tech](http://B.Tech)' → 'B.Tech'"""
    return _MD_LINK_RE.sub(r"\1", text) if text else text

strip_md_udf = F.udf(_strip_md, StringType())

# ─── Schema ───────────────────────────────────────────────────────────────────

RAW_SCHEMA = StructType([
    StructField("id",               IntegerType(), True),
    StructField("year",             IntegerType(), True),
    StructField("institute_type",   StringType(),  True),
    StructField("round_no",         IntegerType(), True),
    StructField("quota",            StringType(),  True),
    StructField("pool",             StringType(),  True),
    StructField("institute_short",  StringType(),  True),
    StructField("program_name",     StringType(),  True),
    StructField("program_duration", StringType(),  True),
    StructField("degree_short",     StringType(),  True),
    StructField("category",         StringType(),  True),
    StructField("opening_rank",     IntegerType(), True),
    StructField("closing_rank",     IntegerType(), True),
    StructField("is_preparatory",   IntegerType(), True),
])

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    spark = (
        SparkSession.builder
        .appName("JEE_Ingest")
        .config("spark.sql.shuffle.partitions", "8")   # right-sized for small data
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # ── Read ─────────────────────────────────────────────────────────────────
    print(f"Reading CSV from {RAW_CSV_PATH} ...")
    df = spark.read.csv(RAW_CSV_PATH, header=True, schema=RAW_SCHEMA)
    raw_count = df.count()
    print(f"  Raw rows : {raw_count:,}")

    # ── Clean ────────────────────────────────────────────────────────────────
    df_clean = (
        df
        # 1. Must-have columns
        .dropna(subset=[
            "year", "quota", "pool", "category",
            "closing_rank", "opening_rank",
            "institute_short", "program_name", "institute_type",
        ])

        # 2. Rank sanity
        .filter(F.col("closing_rank") > 0)
        .filter(F.col("opening_rank") > 0)
        .filter(F.col("closing_rank") >= F.col("opening_rank"))

        # 3. Final allocation rounds only
        .filter(F.col("round_no").isin(FINAL_ROUNDS))

        # 4. No preparatory seats
        .filter(F.col("is_preparatory") == 0)

        # 5. Normalise strings
        .withColumn("quota",           F.upper(F.trim(F.col("quota"))))
        .withColumn("pool",            F.trim(F.col("pool")))
        .withColumn("category",        F.upper(F.trim(F.col("category"))))
        .withColumn("institute_type",  F.upper(F.trim(F.col("institute_type"))))
        .withColumn("institute_short", F.trim(F.col("institute_short")))
        .withColumn("program_name",    F.trim(F.col("program_name")))
        .withColumn("degree_short",    strip_md_udf(F.trim(F.col("degree_short"))))

        # 6. Canonical pool values
        .withColumn(
            "pool",
            F.when(F.lower(F.col("pool")).contains("female"), F.lit("Female-Only"))
             .otherwise(F.lit("Gender-Neutral"))
        )

        # 7. Canonical category aliases
        .withColumn(
            "category",
            F.when(F.col("category") == "EWS",  F.lit("GEN-EWS"))
             .when(F.col("category") == "OPEN", F.lit("GEN"))
             .otherwise(F.col("category"))
        )
    )

    clean_count = df_clean.count()
    print(f"  Clean rows : {clean_count:,}  (dropped {raw_count - clean_count:,})")

    # ── Write ────────────────────────────────────────────────────────────────
    print(f"Writing parquet to {CLEAN_PARQUET} ...")
    (
        df_clean.write
        .mode("overwrite")
        .partitionBy("year", "institute_type", "category")
        .parquet(CLEAN_PARQUET)
    )

    # Quick summary
    print("\nPartition counts (year × institute_type × category):")
    df_clean.groupBy("year", "institute_type", "category") \
            .count() \
            .orderBy("year", "institute_type", "category") \
            .show(50, truncate=False)

    print(f"\nIngestion complete → {CLEAN_PARQUET}")
    spark.stop()


if __name__ == "__main__":
    main()