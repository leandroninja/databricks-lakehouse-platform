"""Camada Gold: agregações prontas para o negócio e tabelas de feature store."""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from delta.tables import DeltaTable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def build_daily_revenue_summary(spark: SparkSession, silver_table: str, gold_table: str) -> DataFrame:
    """Agrega métricas diárias de receita por produto e região."""
    df = spark.table(silver_table)

    gold = (
        df.filter(F.col("_is_current") == True)
        .groupBy("transaction_date", "product_id", "region", "channel")
        .agg(
            F.sum("amount").alias("total_revenue"),
            F.count("transaction_id").alias("transaction_count"),
            F.avg("amount").alias("avg_transaction_value"),
            F.countDistinct("customer_id").alias("unique_customers"),
            F.sum(F.when(F.col("status") == "refunded", F.col("amount")).otherwise(0)).alias("refund_amount"),
            F.sum(F.when(F.col("status") == "completed", 1).otherwise(0)).alias("completed_count"),
        )
        .withColumn("completion_rate", F.col("completed_count") / F.col("transaction_count"))
        .withColumn("net_revenue", F.col("total_revenue") - F.col("refund_amount"))
        .withColumn("_gold_updated_at", F.current_timestamp())
    )

    _write_gold(gold, gold_table)
    logger.info(f"Gold: daily_revenue_summary written — {gold.count()} rows")
    return gold


def build_customer_360(spark: SparkSession, silver_transactions: str, silver_customers: str, gold_table: str) -> DataFrame:
    """Constrói a visão 360 do cliente com pontuação RFM."""
    txns = spark.table(silver_transactions).filter(F.col("_is_current") == True)
    customers = spark.table(silver_customers).filter(F.col("_is_current") == True)

    snapshot_date = datetime.utcnow().date()

    rfm = (
        txns.groupBy("customer_id")
        .agg(
            F.max("transaction_date").alias("last_purchase_date"),
            F.count("transaction_id").alias("frequency"),
            F.sum("amount").alias("monetary_value"),
        )
        .withColumn(
            "recency_days",
            F.datediff(F.lit(snapshot_date), F.col("last_purchase_date"))
        )
    )

    # Pontuação RFM baseada em quartis (1=pior, 4=melhor)
    r_window = Window.orderBy(F.col("recency_days"))
    f_window = Window.orderBy(F.col("frequency"))
    m_window = Window.orderBy(F.col("monetary_value"))

    rfm_scored = (
        rfm
        .withColumn("r_score", F.ntile(4).over(r_window.desc()))   # menor recência = melhor
        .withColumn("f_score", F.ntile(4).over(f_window))
        .withColumn("m_score", F.ntile(4).over(m_window))
        .withColumn("rfm_score", F.col("r_score") + F.col("f_score") + F.col("m_score"))
        .withColumn(
            "customer_segment",
            F.when(F.col("rfm_score") >= 10, "Champions")
            .when(F.col("rfm_score") >= 8, "Loyal Customers")
            .when(F.col("rfm_score") >= 6, "Potential Loyalists")
            .when(F.col("rfm_score") >= 4, "At Risk")
            .otherwise("Lost"),
        )
    )

    gold = customers.join(rfm_scored, "customer_id", "left").withColumn("_gold_updated_at", F.current_timestamp())
    _write_gold(gold, gold_table)
    logger.info(f"Gold: customer_360 written — {gold.count()} rows")
    return gold


def build_product_performance(spark: SparkSession, silver_table: str, gold_table: str) -> DataFrame:
    """Desempenho por produto com cálculos de crescimento MoM (mês a mês)."""
    df = spark.table(silver_table).filter(F.col("_is_current") == True)

    monthly = (
        df.withColumn("year_month", F.date_format("transaction_date", "yyyy-MM"))
        .groupBy("year_month", "product_id", "category")
        .agg(
            F.sum("amount").alias("monthly_revenue"),
            F.count("transaction_id").alias("monthly_orders"),
            F.countDistinct("customer_id").alias("monthly_buyers"),
        )
    )

    window_prev = Window.partitionBy("product_id").orderBy("year_month")

    gold = (
        monthly
        .withColumn("prev_month_revenue", F.lag("monthly_revenue", 1).over(window_prev))
        .withColumn(
            "mom_growth_pct",
            F.when(
                F.col("prev_month_revenue") > 0,
                ((F.col("monthly_revenue") - F.col("prev_month_revenue")) / F.col("prev_month_revenue") * 100)
            ).otherwise(None)
        )
        .withColumn("_gold_updated_at", F.current_timestamp())
    )

    _write_gold(gold, gold_table)
    logger.info(f"Gold: product_performance written — {gold.count()} rows")
    return gold


def build_operational_kpis(spark: SparkSession, silver_table: str, gold_table: str) -> DataFrame:
    """Tabela de KPIs executivos para dashboards — atualizada diariamente."""
    df = spark.table(silver_table).filter(F.col("_is_current") == True)

    today = datetime.utcnow().date()

    kpis = (
        df.filter(F.col("transaction_date") >= F.date_sub(F.lit(today), 30))
        .agg(
            F.sum("amount").alias("revenue_30d"),
            F.count("transaction_id").alias("orders_30d"),
            F.countDistinct("customer_id").alias("active_customers_30d"),
            F.avg("amount").alias("aov_30d"),
            F.sum(F.when(F.col("status") == "completed", F.col("amount")).otherwise(0)).alias("net_revenue_30d"),
        )
        .withColumn("snapshot_date", F.lit(str(today)))
        .withColumn("_gold_updated_at", F.current_timestamp())
    )

    _write_gold(kpis, gold_table, mode="overwrite")
    logger.info("Gold: operational_kpis refreshed")
    return kpis


def _write_gold(df: DataFrame, table: str, mode: str = "append") -> None:
    """Grava o DataFrame Gold na tabela Delta e executa OPTIMIZE."""
    if mode == "overwrite":
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    else:
        df.write.format("delta").mode("append").saveAsTable(table)

    # OPTIMIZE para consultas rápidas no Grafana
    spark = df.sparkSession
    spark.sql(f"OPTIMIZE {table}")
