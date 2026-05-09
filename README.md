# Plataforma Databricks Lakehouse

![Databricks](https://img.shields.io/badge/Databricks-Lakehouse-FF3621?logo=databricks)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-3.0+-00ADD8)
![MLflow](https://img.shields.io/badge/MLflow-2.10+-0194E2?logo=mlflow)
![PySpark](https://img.shields.io/badge/PySpark-3.5+-E25A1C?logo=apachespark)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)
![Licença](https://img.shields.io/badge/Licença-MIT-green)

> Plataforma **Lakehouse de nível enterprise** sobre Azure Databricks com arquitetura Medalha (Bronze/Silver/Gold), Auto Loader com evolução de schema, SCD Tipo 2 via Delta MERGE, treinamento de modelos com MLflow + Hyperopt e qualidade de dados com Great Expectations.

---

## Arquitetura Medalha

```
┌─────────────────────────────────────────────────────────────────┐
│                    FONTES DE DADOS                               │
│  S3 / ADLS / Kafka / APIs REST / Bancos de Dados                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │     CAMADA BRONZE        │
              │  Auto Loader (cloudFiles)│
              │  Schema Evolution        │
              │  Quarentena de erros     │
              │  Metadados: hash, fonte  │
              └────────────┬────────────┘
                           │ MERGE / UPSERT
              ┌────────────▼────────────┐
              │     CAMADA SILVER        │
              │  SCD Tipo 2 (Delta MERGE)│
              │  Deduplicação (Window)   │
              │  Padronização de tipos   │
              │  Watermark de qualidade  │
              └────────────┬────────────┘
                           │ Agregações
              ┌────────────▼────────────┐
              │      CAMADA GOLD         │
              │  Resumo diário de receita│
              │  Visão 360° do cliente   │
              │  Performance de produto  │
              │  KPIs executivos         │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   Dashboards ML/AI    Relatórios       Feature Store
   (Grafana/PowerBI)  (MLflow)         (Unity Catalog)
```

---

## Funcionalidades

| Funcionalidade | Descrição |
|----------------|-----------|
| Auto Loader | Ingestão incremental com cloudFiles, evolução de schema, quarentena |
| SCD Tipo 2 | Delta MERGE com `_valid_from`, `_valid_to`, `_is_current`, `_checksum` |
| ML com MLflow | XGBoost + busca de hiperparâmetros Bayesiana (Hyperopt TPE) |
| Qualidade de dados | Great Expectations com relatório gravado no Delta |
| Manutenção Delta | OPTIMIZE, VACUUM, Z-ORDER, CLONE, histórico e restore |
| Orquestração | Databricks Workflows com 6 tarefas, clusters dedicados e notificações |
| Governança | Unity Catalog, linhagem de dados, controle de acesso por coluna |
| Camada Gold | RFM scoring, crescimento MoM, KPIs em tempo real |

---

## Estrutura do Projeto

```
databricks-lakehouse-platform/
├── src/
│   ├── ingestion/bronze/
│   │   └── ingest_raw_data.py          # Auto Loader, metadados, quarentena
│   ├── transformation/
│   │   ├── silver/
│   │   │   └── cleanse_transform.py    # Dedup, SCD2 MERGE, padronização
│   │   └── gold/
│   │       └── build_gold_layer.py     # Receita, Cliente 360, KPIs
│   ├── ml/training/
│   │   └── train_model.py              # XGBoost + Hyperopt + MLflow
│   ├── data_quality/expectations/
│   │   └── data_quality_checks.py      # Great Expectations + relatório Delta
│   └── utils/
│       └── delta_utils.py              # OPTIMIZE, VACUUM, CLONE, restore
└── config/jobs/
    └── etl_pipeline_job.json           # Definição completa do Databricks Workflows
```

---

## SLA de Qualidade de Dados

| Expectativa | Camada | Limiar |
|-------------|--------|--------|
| Completude | Silver | ≥ 99% sem nulos em campos-chave |
| Unicidade | Silver | 0% de duplicatas por chave de negócio |
| Frescor | Gold | Dados atualizados a cada 6 horas |
| Precisão | Silver | Formato correto em 100% dos campos tipados |
| Conformidade | Gold | Validação de regex para e-mails e IDs |

---

## Início Rápido

```bash
# Instalar dependências
pip install databricks-sdk delta-spark mlflow great-expectations hyperopt xgboost

# Configurar acesso ao Databricks
export DATABRICKS_HOST="https://adb-xxx.azuredatabricks.net"
export DATABRICKS_TOKEN="dapi..."

# Executar ingestão Bronze localmente (modo dev)
python -m src.ingestion.bronze.ingest_raw_data

# Treinar modelo com MLflow
python -m src.ml.training.train_model

# Fazer deploy do job completo via Databricks CLI
databricks jobs create --json @config/jobs/etl_pipeline_job.json
```

---

## Autor

**Leandro Oliveira Moraes**  
Arquiteto Sênior DevOps & Multi-Cloud | FinOps & Dados  
Databricks Certified | Intel Cloud FinOps Certified  
[LinkedIn](https://linkedin.com/in/leandro-oliveira-26b14768)
