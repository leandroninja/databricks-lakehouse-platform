"""Verificações de qualidade de dados usando Great Expectations e expectativas do Delta Live Tables."""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
import great_expectations as gx
from great_expectations.dataset import SparkDFDataset
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class DeltaLakeDataQualityChecker:
    """Framework abrangente de validação de qualidade de dados para tabelas Delta."""

    def __init__(self, spark: SparkSession, catalog: str, schema: str):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.results = []

    def validate_table(
        self,
        table_name: str,
        expectations: list[dict],
        fail_on_error: bool = False,
    ) -> dict:
        """Executa um conjunto de expectations em uma tabela Delta."""
        full_table = f"{self.catalog}.{self.schema}.{table_name}"
        logger.info(f"Running quality checks on: {full_table}")

        df = self.spark.table(full_table)
        ge_dataset = SparkDFDataset(df)

        suite_results = {
            "table": full_table,
            "run_timestamp": datetime.utcnow().isoformat(),
            "total_rows": df.count(),
            "checks": [],
        }

        passed = 0
        failed = 0

        for exp in expectations:
            method = exp["expectation"]
            kwargs = {k: v for k, v in exp.items() if k != "expectation"}

            try:
                result = getattr(ge_dataset, method)(**kwargs)
                success = result["success"]
                check_result = {
                    "expectation": method,
                    "kwargs": kwargs,
                    "success": success,
                    "result": result.get("result", {}),
                }
                suite_results["checks"].append(check_result)

                if success:
                    passed += 1
                else:
                    failed += 1
                    logger.warning(f"Quality check FAILED: {method}({kwargs})")
                    if fail_on_error:
                        raise ValueError(f"Data quality gate failed: {method}")

            except AttributeError:
                logger.error(f"Unknown expectation: {method}")

        suite_results["passed"] = passed
        suite_results["failed"] = failed
        suite_results["pass_rate"] = passed / (passed + failed) if (passed + failed) > 0 else 0

        self._write_quality_report(suite_results)
        self.results.append(suite_results)

        logger.info(f"Quality check complete: {passed}/{passed+failed} passed ({suite_results['pass_rate']:.1%})")
        return suite_results

    def _write_quality_report(self, report: dict):
        """Persiste o relatório de qualidade em tabela Delta para dashboards de monitoramento."""
        report_df = self.spark.createDataFrame([{
            "table_name": report["table"],
            "run_timestamp": report["run_timestamp"],
            "total_rows": report["total_rows"],
            "checks_passed": report["passed"],
            "checks_failed": report["failed"],
            "pass_rate": report["pass_rate"],
            "details": json.dumps(report["checks"]),
        }])

        (
            report_df.write.format("delta")
            .mode("append")
            .saveAsTable(f"{self.catalog}.monitoring.data_quality_runs")
        )

    def check_freshness(self, table_name: str, ts_column: str, max_lag_minutes: int = 60) -> bool:
        """Verifica o SLA de atualidade dos dados."""
        full_table = f"{self.catalog}.{self.schema}.{table_name}"
        max_ts = self.spark.table(full_table).select(F.max(ts_column)).collect()[0][0]

        if max_ts is None:
            logger.error(f"Table {full_table} has no data!")
            return False

        lag_minutes = (datetime.utcnow() - max_ts.replace(tzinfo=None)).total_seconds() / 60
        is_fresh = lag_minutes <= max_lag_minutes

        if not is_fresh:
            logger.warning(f"Freshness SLA VIOLATED: {full_table} lag={lag_minutes:.1f}min (max={max_lag_minutes}min)")
        else:
            logger.info(f"Freshness OK: {full_table} lag={lag_minutes:.1f}min")

        return is_fresh

    def profile_table(self, table_name: str) -> DataFrame:
        """Gera perfil estatístico para todas as colunas numéricas."""
        full_table = f"{self.catalog}.{self.schema}.{table_name}"
        df = self.spark.table(full_table)

        numeric_cols = [f.name for f in df.schema.fields
                        if f.dataType.__class__.__name__ in ("LongType", "DoubleType", "IntegerType", "DecimalType")]

        if not numeric_cols:
            logger.warning("Nenhuma coluna numérica encontrada para perfilamento")
            return self.spark.createDataFrame([], schema="col STRING")

        profile_exprs = []
        for col in numeric_cols:
            profile_exprs.extend([
                F.count(col).alias(f"{col}_count"),
                F.count_if(F.col(col).isNull()).alias(f"{col}_nulls"),
                F.min(col).alias(f"{col}_min"),
                F.max(col).alias(f"{col}_max"),
                F.mean(col).alias(f"{col}_mean"),
                F.stddev(col).alias(f"{col}_stddev"),
                F.percentile_approx(col, 0.5).alias(f"{col}_p50"),
                F.percentile_approx(col, 0.95).alias(f"{col}_p95"),
            ])

        return df.select(F.lit(table_name).alias("table"), *profile_exprs)


def build_silver_expectations(table_name: str) -> list[dict]:
    """Suite padrão de expectations para a camada Silver."""
    return [
        {"expectation": "expect_column_to_exist", "column": "id"},
        {"expectation": "expect_column_values_to_not_be_null", "column": "id"},
        {"expectation": "expect_column_values_to_be_unique", "column": "id"},
        {"expectation": "expect_column_to_exist", "column": "event_time"},
        {"expectation": "expect_column_values_to_not_be_null", "column": "event_time"},
        {"expectation": "expect_column_values_to_be_between", "column": "amount", "min_value": 0, "max_value": 1_000_000},
        {"expectation": "expect_table_row_count_to_be_between", "min_value": 1, "max_value": 100_000_000},
        {"expectation": "expect_column_proportion_of_unique_values_to_be_between", "column": "id", "min_value": 0.99},
    ]


if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()

    checker = DeltaLakeDataQualityChecker(spark, catalog="prod_catalog", schema="silver")

    results = checker.validate_table(
        table_name="events_silver",
        expectations=build_silver_expectations("events_silver"),
        fail_on_error=True,
    )

    is_fresh = checker.check_freshness(
        table_name="events_silver",
        ts_column="event_time",
        max_lag_minutes=30,
    )

    print(f"Quality: {results['pass_rate']:.1%} pass rate | Fresh: {is_fresh}")
