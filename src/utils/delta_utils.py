"""Operações Delta Lake: OPTIMIZE, VACUUM, Z-Order, time travel, clonagem."""
from pyspark.sql import SparkSession
from delta.tables import DeltaTable
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def optimize_table(spark: SparkSession, table_name: str, z_order_cols: list[str] = None) -> dict:
    """Executa OPTIMIZE com Z-ORDER opcional para melhorar o desempenho de consultas."""
    logger.info(f"Optimizing table: {table_name}" + (f" ZORDER BY {z_order_cols}" if z_order_cols else ""))

    if z_order_cols:
        cols = ", ".join(z_order_cols)
        result = spark.sql(f"OPTIMIZE {table_name} ZORDER BY ({cols})")
    else:
        result = spark.sql(f"OPTIMIZE {table_name}")

    metrics = result.collect()[0].asDict() if result.count() > 0 else {}
    logger.info(f"Optimize complete: {metrics}")
    return metrics


def vacuum_table(spark: SparkSession, table_name: str, retain_hours: int = 168) -> None:
    """Remove arquivos Delta antigos mantendo o histórico pelo número de horas especificado."""
    logger.info(f"Vacuuming {table_name} (retain {retain_hours}h)")
    spark.sql(f"VACUUM {table_name} RETAIN {retain_hours} HOURS")
    logger.info(f"Vacuum complete: {table_name}")


def get_table_history(spark: SparkSession, table_name: str, limit: int = 10) -> "DataFrame":
    """Retorna o histórico de transações da tabela Delta."""
    return spark.sql(f"DESCRIBE HISTORY {table_name} LIMIT {limit}")


def restore_table(spark: SparkSession, table_name: str, version: int = None, timestamp: str = None) -> None:
    """Restaura a tabela Delta para uma versão ou timestamp anterior."""
    if version is not None:
        spark.sql(f"RESTORE TABLE {table_name} TO VERSION AS OF {version}")
        logger.info(f"Restored {table_name} to version {version}")
    elif timestamp:
        spark.sql(f"RESTORE TABLE {table_name} TO TIMESTAMP AS OF '{timestamp}'")
        logger.info(f"Restored {table_name} to {timestamp}")
    else:
        raise ValueError("É necessário especificar version ou timestamp para restauração")


def clone_table(
    spark: SparkSession,
    source_table: str,
    target_table: str,
    deep: bool = False,
) -> None:
    """Clona uma tabela Delta (shallow ou deep) para testes/backups."""
    clone_type = "DEEP" if deep else "SHALLOW"
    spark.sql(f"CREATE OR REPLACE TABLE {target_table} {clone_type} CLONE {source_table}")
    logger.info(f"{clone_type} clone: {source_table} -> {target_table}")


def get_table_stats(spark: SparkSession, table_name: str) -> dict:
    """Obtém estatísticas completas da tabela."""
    detail = spark.sql(f"DESCRIBE DETAIL {table_name}").collect()[0].asDict()
    return {
        "table": table_name,
        "format": detail.get("format"),
        "location": detail.get("location"),
        "created_at": str(detail.get("createdAt")),
        "last_modified": str(detail.get("lastModified")),
        "num_files": detail.get("numFiles"),
        "size_in_bytes": detail.get("sizeInBytes"),
        "size_gb": round(detail.get("sizeInBytes", 0) / (1024**3), 2),
        "num_partitions": detail.get("numPartitions"),
        "properties": detail.get("properties"),
    }


def run_maintenance(
    spark: SparkSession,
    catalog: str,
    schema: str,
    tables_config: dict,
):
    """
    Executa manutenção (OPTIMIZE + VACUUM) em múltiplas tabelas.
    tables_config: {table_name: {"z_order": [...], "retain_hours": int}}
    """
    for table_name, config in tables_config.items():
        full_name = f"{catalog}.{schema}.{table_name}"
        try:
            optimize_table(spark, full_name, config.get("z_order"))
            vacuum_table(spark, full_name, config.get("retain_hours", 168))
            stats = get_table_stats(spark, full_name)
            logger.info(f"Maintenance complete: {full_name} | Size: {stats['size_gb']}GB | Files: {stats['num_files']}")
        except Exception as e:
            logger.error(f"Maintenance failed for {full_name}: {e}")
            raise


if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()

    TABLES_CONFIG = {
        "events_silver": {
            "z_order": ["customer_id", "event_time"],
            "retain_hours": 168,
        },
        "transactions_gold": {
            "z_order": ["merchant_id", "transaction_date"],
            "retain_hours": 720,
        },
    }

    run_maintenance(spark, "prod_catalog", "silver", TABLES_CONFIG)
