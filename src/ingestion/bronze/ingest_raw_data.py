"""Ingestão na camada Bronze usando Auto Loader para ingestão de arquivos incremental e escalável."""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType
from delta.tables import DeltaTable
import logging

logger = logging.getLogger(__name__)


def get_spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


def build_autoloader_stream(
    spark: SparkSession,
    source_path: str,
    schema_location: str,
    file_format: str = "json",
    schema: StructType = None,
) -> "DataFrame":
    """Configura o stream do Auto Loader com inferência e evolução de schema."""
    reader = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", file_format)
        .option("cloudFiles.schemaLocation", schema_location)
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.maxFilesPerTrigger", 10000)
        .option("cloudFiles.backfillInterval", "1 day")
        .option("recursiveFileLookup", "true")
    )

    if schema:
        reader = reader.schema(schema)

    return reader.load(source_path)


def add_metadata_columns(df: "DataFrame", source_name: str) -> "DataFrame":
    """Enriquece dados brutos com metadados de ingestão para rastreabilidade e depuração."""
    return df.withColumn("_ingestion_timestamp", F.current_timestamp()) \
             .withColumn("_source_file", F.col("_metadata.file_path")) \
             .withColumn("_source_name", F.lit(source_name)) \
             .withColumn("_ingestion_date", F.current_date()) \
             .withColumn("_record_hash", F.sha2(F.to_json(F.struct(*df.columns)), 256))


def quarantine_bad_records(df: "DataFrame", rules: list[tuple]) -> tuple:
    """
    Separa registros válidos de inválidos usando regras configuráveis.
    Retorna (valid_df, quarantine_df).
    """
    quarantine_conditions = []
    for col_name, rule_expr in rules:
        quarantine_conditions.append(~F.expr(rule_expr).alias(f"fail_{col_name}"))

    if not quarantine_conditions:
        return df, df.filter(F.lit(False))

    combined_invalid = quarantine_conditions[0]
    for cond in quarantine_conditions[1:]:
        combined_invalid = combined_invalid | cond

    valid_df = df.filter(~combined_invalid)
    quarantine_df = df.filter(combined_invalid) \
                      .withColumn("_quarantine_reason", F.lit("Failed validation rules")) \
                      .withColumn("_quarantine_timestamp", F.current_timestamp())

    return valid_df, quarantine_df


def write_bronze_stream(
    df: "DataFrame",
    target_table: str,
    checkpoint_path: str,
    trigger_interval: str = "5 minutes",
    partition_cols: list[str] = None,
) -> "StreamingQuery":
    """Grava o stream do Auto Loader na tabela Delta da camada Bronze com checkpointing."""
    writer = (
        df.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", checkpoint_path)
        .option("mergeSchema", "true")
        .trigger(processingTime=trigger_interval)
    )

    if partition_cols:
        writer = writer.partitionBy(*partition_cols)

    return writer.toTable(target_table)


def run_bronze_ingestion(
    source_path: str,
    target_catalog: str,
    target_schema: str,
    source_name: str,
    file_format: str = "json",
    validation_rules: list[tuple] = None,
):
    """Orquestra o pipeline de ingestão da camada Bronze."""
    spark = get_spark()

    schema_location = f"abfss://schemas@{target_catalog}.dfs.core.windows.net/{source_name}"
    checkpoint_path = f"abfss://checkpoints@{target_catalog}.dfs.core.windows.net/{source_name}/bronze"
    target_table = f"{target_catalog}.{target_schema}.{source_name}_bronze"
    quarantine_table = f"{target_catalog}.{target_schema}.{source_name}_quarantine"

    logger.info(f"Starting Bronze ingestion: {source_name} -> {target_table}")

    raw_stream = build_autoloader_stream(
        spark=spark,
        source_path=source_path,
        schema_location=schema_location,
        file_format=file_format,
    )

    enriched_stream = add_metadata_columns(raw_stream, source_name)

    if validation_rules:
        valid_stream, quarantine_stream = quarantine_bad_records(enriched_stream, validation_rules)

        quarantine_query = write_bronze_stream(
            df=quarantine_stream,
            target_table=quarantine_table,
            checkpoint_path=f"{checkpoint_path}_quarantine",
            partition_cols=["_ingestion_date"],
        )
        logger.info(f"Quarantine stream started: {quarantine_table}")
    else:
        valid_stream = enriched_stream

    main_query = write_bronze_stream(
        df=valid_stream,
        target_table=target_table,
        checkpoint_path=checkpoint_path,
        partition_cols=["_ingestion_date"],
    )

    logger.info(f"Bronze stream started: {target_table} | Query ID: {main_query.id}")
    return main_query


if __name__ == "__main__":
    import sys

    validation_rules = [
        ("id", "id IS NOT NULL AND id != ''"),
        ("event_time", "event_time IS NOT NULL"),
        ("amount", "amount >= 0"),
    ]

    query = run_bronze_ingestion(
        source_path=sys.argv[1] if len(sys.argv) > 1 else "abfss://landing@storageaccount.dfs.core.windows.net/events/",
        target_catalog="prod_catalog",
        target_schema="bronze",
        source_name="events",
        file_format="json",
        validation_rules=validation_rules,
    )

    query.awaitTermination()
