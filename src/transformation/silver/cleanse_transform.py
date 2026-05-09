"""Camada Silver: deduplicação, SCD Tipo 2 e transformações de qualidade de dados."""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import TimestampType
from delta.tables import DeltaTable
import logging

logger = logging.getLogger(__name__)


def get_spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


def deduplicate(df: "DataFrame", key_cols: list[str], order_col: str = "_ingestion_timestamp") -> "DataFrame":
    """Mantém apenas o registro mais recente por chave, ordenado por order_col."""
    window = Window.partitionBy(*key_cols).orderBy(F.col(order_col).desc())
    return (
        df.withColumn("_row_num", F.row_number().over(window))
          .filter(F.col("_row_num") == 1)
          .drop("_row_num")
    )


def standardize_types(df: "DataFrame") -> "DataFrame":
    """Converte e padroniza tipos de dados comuns."""
    result = df

    for field in df.schema.fields:
        if "date" in field.name.lower() and field.dataType == StringType():
            result = result.withColumn(
                field.name,
                F.to_timestamp(F.col(field.name), "yyyy-MM-dd HH:mm:ss")
            )
        if "amount" in field.name.lower() or "value" in field.name.lower():
            result = result.withColumn(field.name, F.col(field.name).cast("decimal(18,2)"))
        if "email" in field.name.lower():
            result = result.withColumn(field.name, F.lower(F.trim(F.col(field.name))))

    return result


def apply_scd2_merge(
    spark: SparkSession,
    source_df: "DataFrame",
    target_table: str,
    key_cols: list[str],
    change_cols: list[str],
) -> dict:
    """
    Implementa SCD Tipo 2 via Delta MERGE.
    Rastreia o histórico de linhas com as colunas _valid_from/_valid_to/is_current.
    """
    if not DeltaTable.isDeltaTable(spark, target_table.replace(".", "/")):
        logger.info(f"Tabela destino {target_table} não existe — criando com schema SCD2")
        (
            source_df
            .withColumn("_valid_from", F.current_timestamp())
            .withColumn("_valid_to", F.lit(None).cast(TimestampType()))
            .withColumn("_is_current", F.lit(True))
            .withColumn("_checksum", F.sha2(F.to_json(F.struct(*change_cols)), 256))
        ).write.format("delta").saveAsTable(target_table)
        return {"inserted": source_df.count(), "updated": 0, "closed": 0}

    target = DeltaTable.forName(spark, target_table)

    source_with_checksum = source_df.withColumn(
        "_new_checksum", F.sha2(F.to_json(F.struct(*change_cols)), 256)
    )

    join_condition = " AND ".join([f"target.{c} = source.{c}" for c in key_cols])
    merge_condition = f"({join_condition}) AND target._is_current = true"

    (
        target.alias("target")
        .merge(source_with_checksum.alias("source"), merge_condition)
        .whenMatchedUpdate(
            condition="target._checksum != source._new_checksum",
            set={
                "_is_current": "false",
                "_valid_to": "current_timestamp()",
            }
        )
        .whenNotMatchedInsert(
            values={
                **{c: f"source.{c}" for c in source_df.columns},
                "_valid_from": "current_timestamp()",
                "_valid_to": "null",
                "_is_current": "true",
                "_checksum": "source._new_checksum",
            }
        )
        .execute()
    )

    stats = {"table": target_table, "status": "scd2_merge_completed"}
    logger.info(f"SCD2 merge completed: {stats}")
    return stats


def run_silver_transformation(
    source_catalog: str,
    source_schema: str,
    source_table: str,
    target_catalog: str,
    target_schema: str,
    key_cols: list[str],
    change_cols: list[str],
    batch_size: int = 100_000,
):
    """Executa o pipeline completo de transformação da camada Silver."""
    spark = get_spark()

    source_full = f"{source_catalog}.{source_schema}.{source_table}"
    target_full = f"{target_catalog}.{target_schema}.{source_table.replace('_bronze','_silver')}"

    logger.info(f"Silver transformation: {source_full} -> {target_full}")

    watermark_key = "_ingestion_timestamp"
    last_processed = _get_watermark(spark, target_full, watermark_key)

    source_df = (
        spark.table(source_full)
        .filter(F.col(watermark_key) > last_processed)
        .limit(batch_size)
    )

    record_count = source_df.count()
    if record_count == 0:
        logger.info("Nenhum registro novo para processar")
        return

    logger.info(f"Processando {record_count} novos registros")

    deduped_df = deduplicate(source_df, key_cols, watermark_key)
    typed_df = standardize_types(deduped_df)
    clean_df = typed_df.filter(
        F.col("id").isNotNull() &
        (F.col("id") != "") &
        ~F.col("_ingestion_timestamp").isNull()
    )

    stats = apply_scd2_merge(
        spark=spark,
        source_df=clean_df,
        target_table=target_full,
        key_cols=key_cols,
        change_cols=change_cols,
    )

    _update_watermark(spark, target_full, watermark_key, source_df)
    logger.info(f"Silver transformation complete: {stats}")


def _get_watermark(spark, table_name: str, ts_col: str) -> "Timestamp":
    try:
        return spark.sql(f"SELECT MAX({ts_col}) FROM {table_name}").collect()[0][0] or "1970-01-01"
    except Exception:
        return "1970-01-01"


def _update_watermark(spark, table_name: str, ts_col: str, df: "DataFrame"):
    max_ts = df.select(F.max(ts_col)).collect()[0][0]
    logger.info(f"New watermark for {table_name}: {max_ts}")


if __name__ == "__main__":
    run_silver_transformation(
        source_catalog="prod_catalog",
        source_schema="bronze",
        source_table="events_bronze",
        target_catalog="prod_catalog",
        target_schema="silver",
        key_cols=["id"],
        change_cols=["status", "amount", "customer_id", "updated_at"],
    )
