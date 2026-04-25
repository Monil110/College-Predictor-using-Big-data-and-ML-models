"""
jee_features.py
─────────────────────────────────────────────────────────────────────────────
Stage 2: Build model-ready feature rows from cleaned parquet.

Reads  : /app/data/processed/jee_cleaned
Writes :
  /app/data/processed/jee_features/iit   (institute_type == IIT)
  /app/data/processed/jee_features/nit   (all others)

Features generated
──────────────────
Per-year base (closing_rank_year, opening_rank_year, rank_spread_year)
Multi-year agg:
  closing_rank_max    – worst (highest) closing rank seen across years
  opening_rank_min    – best (lowest) opening rank seen across years
  closing_rank_avg    – mean closing rank across years
  closing_rank_std    – std dev (0 when only one year available)
  rank_spread_avg     – mean (closing – opening) across years
  years_available     – how many years of data exist for this row
Trend:
  yoy_rank_change     – closing rank latest year minus previous year
                        positive = harder to enter, negative = easier
  trend_direction     – UP / DOWN / STABLE / UNKNOWN
Competition signal:
  rank_pressure       – closing_rank_avg / opening_rank_min  (≥1 always)
  difficulty_pct      – percentile within (institute_type, category, quota, pool)
                        0 = easiest programme, 100 = hardest

Run:
    docker exec spark-master \
        /opt/spark/bin/spark-submit \
        --master spark://spark-master:7077 \
        /app/jee_features.py
─────────────────────────────────────────────────────────────────────────────
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

# ─── Paths ────────────────────────────────────────────────────────────────────

CLEAN_PARQUET = "/app/data/processed/jee_cleaned"
OUT_IIT       = "/app/data/processed/jee_features/iit"
OUT_NIT       = "/app/data/processed/jee_features/nit"

# Columns that identify a unique (institute × programme × seat-type) combination
GROUP_COLS = [
    "institute_type",
    "institute_short",
    "program_name",
    "degree_short",
    "category",
    "quota",
    "pool",
]

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    spark = (
        SparkSession.builder
        .appName("JEE_Features")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading cleaned data from {CLEAN_PARQUET} ...")
    df = spark.read.parquet(CLEAN_PARQUET)
    df.cache()
    print(f"  Rows loaded : {df.count():,}")

    # ── Step 1 : per-year base aggregates ────────────────────────────────────
    per_year = (
        df.groupBy(*GROUP_COLS, "year")
        .agg(
            F.max("closing_rank").alias("closing_rank_year"),
            F.min("opening_rank").alias("opening_rank_year"),
            (F.max("closing_rank") - F.min("opening_rank")).alias("rank_spread_year"),
        )
    )

    # ── Step 2 : multi-year aggregates ───────────────────────────────────────
    multi_year = (
        per_year.groupBy(*GROUP_COLS)
        .agg(
            F.max("closing_rank_year").alias("closing_rank_max"),
            F.min("opening_rank_year").alias("opening_rank_min"),
            F.avg("closing_rank_year").alias("closing_rank_avg"),
            F.stddev("closing_rank_year").alias("closing_rank_std"),
            F.avg("rank_spread_year").alias("rank_spread_avg"),
            F.max("year").alias("latest_year"),
            F.count("year").alias("years_available"),
        )
        .fillna({"closing_rank_std": 0.0, "rank_spread_avg": 0.0})
    )

    # ── Step 3 : year-over-year trend ────────────────────────────────────────
    w_desc = Window.partitionBy(*GROUP_COLS).orderBy(F.col("year").desc())
    ranked = per_year.withColumn("yr_rank", F.row_number().over(w_desc))

    latest = (
        ranked.filter(F.col("yr_rank") == 1)
        .select(*GROUP_COLS, F.col("closing_rank_year").alias("cr_latest"))
    )
    prev = (
        ranked.filter(F.col("yr_rank") == 2)
        .select(*GROUP_COLS, F.col("closing_rank_year").alias("cr_prev"))
    )

    trend = (
        latest.join(prev, on=GROUP_COLS, how="left")
        .withColumn(
            "yoy_rank_change",
            F.when(F.col("cr_prev").isNull(), F.lit(None))
             .otherwise(F.col("cr_latest") - F.col("cr_prev"))
        )
        .withColumn(
            "trend_direction",
            F.when(F.col("cr_prev").isNull(),          F.lit("UNKNOWN"))
             .when(F.col("yoy_rank_change") >  50,     F.lit("UP"))
             .when(F.col("yoy_rank_change") < -50,     F.lit("DOWN"))
             .otherwise(                               F.lit("STABLE"))
        )
        .drop("cr_latest", "cr_prev")
    )

    # ── Step 4 : combine ─────────────────────────────────────────────────────
    features = multi_year.join(trend, on=GROUP_COLS, how="left")

    # ── Step 5 : rank_pressure ───────────────────────────────────────────────
    features = features.withColumn(
        "rank_pressure",
        F.when(
            F.col("opening_rank_min") > 0,
            F.round(F.col("closing_rank_avg") / F.col("opening_rank_min"), 3),
        ).otherwise(F.lit(None))
    )

    # ── Step 6 : difficulty_pct within cohort ────────────────────────────────
    w_cohort = Window.partitionBy("institute_type", "category", "quota", "pool")
    features = features.withColumn(
        "difficulty_pct",
        F.round(
            F.percent_rank().over(w_cohort.orderBy(F.col("closing_rank_avg"))) * 100,
            1,
        )
    )

    features.cache()
    total = features.count()
    print(f"  Total feature rows : {total:,}")

    # ── Step 7 : split IIT / NIT and write ───────────────────────────────────
    iit = features.filter(F.col("institute_type") == "IIT")
    nit = features.filter(F.col("institute_type") != "IIT")

    iit_count = iit.count()
    nit_count = nit.count()

    print(f"\nWriting IIT features ({iit_count:,} rows) → {OUT_IIT}")
    (
        iit.write
        .mode("overwrite")
        .partitionBy("category", "quota")
        .parquet(OUT_IIT)
    )

    print(f"Writing NIT features ({nit_count:,} rows) → {OUT_NIT}")
    (
        nit.write
        .mode("overwrite")
        .partitionBy("category", "quota")
        .parquet(OUT_NIT)
    )

    # Quick sanity print
    print("\nSample IIT features:")
    iit.select(
        "institute_short", "program_name", "category", "quota",
        "closing_rank_max", "closing_rank_avg", "trend_direction", "difficulty_pct"
    ).show(5, truncate=True)

    print("\nFeature engineering complete.")
    print(f"  IIT → {OUT_IIT}")
    print(f"  NIT → {OUT_NIT}")

    df.unpersist()
    features.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()