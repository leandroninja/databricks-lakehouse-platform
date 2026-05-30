"""Schema registry para validacao de schemas Delta Lake entre camadas."""

SCHEMAS = {
    "bronze_raw": {
        "ingestion_date": "timestamp",
        "source_system": "string",
        "raw_payload": "string",
        "checksum": "string",
    },
    "silver_cleansed": {
        "id": "string",
        "created_at": "timestamp",
        "updated_at": "timestamp",
        "is_valid": "boolean",
    },
    "gold_aggregated": {
        "date": "date",
        "metric_name": "string",
        "metric_value": "double",
        "dimensions": "map<string,string>",
    },
}


def get_schema(layer: str) -> dict:
    """Retorna o schema esperado para uma camada do lakehouse."""
    return SCHEMAS.get(layer, {})


def validate_columns(df_columns: list, layer: str) -> list:
    """Retorna colunas faltando em relacao ao schema esperado."""
    expected = set(get_schema(layer).keys())
    actual = set(df_columns)
    return list(expected - actual)
