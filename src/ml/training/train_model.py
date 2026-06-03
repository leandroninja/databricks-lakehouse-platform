"""Rastreamento de experimentos MLflow com ajuste de hiperparâmetros e integração ao model registry."""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging
import os

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "databricks")
EXPERIMENT_NAME = "/Shared/MLOps/fraud-detection"
MODEL_NAME = "fraud-detection-xgboost"


def load_training_data(spark, catalog: str, schema: str, table: str) -> pd.DataFrame:
    """Carrega dados de treinamento da tabela Delta da camada Gold."""
    df = spark.table(f"{catalog}.{schema}.{table}") \
              .filter("_is_current = true") \
              .toPandas()
    logger.info(f"Loaded {len(df)} training records from {catalog}.{schema}.{table}")
    return df


def prepare_features(df: pd.DataFrame, target_col: str, feature_cols: list) -> tuple:
    """Prepara a matriz de features e o vetor alvo com normalização."""
    X = df[feature_cols].fillna(0)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def compute_metrics(y_true, y_pred, y_pred_proba) -> dict:
    """Calcula métricas abrangentes de avaliação do modelo."""
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "precision":   precision_score(y_true, y_pred, average="weighted"),
        "recall":      recall_score(y_true, y_pred, average="weighted"),
        "f1":          f1_score(y_true, y_pred, average="weighted"),
        "roc_auc":     roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="weighted"),
    }


def train_xgboost(params: dict, X_train, y_train, X_test, y_test) -> tuple:
    """Treina o modelo XGBoost com os hiperparâmetros fornecidos."""
    model = XGBClassifier(
        **params,
        use_label_encoder=False,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=20,
        verbose=False,
    )
    return model


def hyperparameter_search(
    X_train, y_train, X_test, y_test,
    max_evals: int = 30,
) -> dict:
    """Otimização bayesiana de hiperparâmetros com Hyperopt."""
    search_space = {
        "n_estimators":   hp.choice("n_estimators", [100, 200, 300, 500]),
        "max_depth":      hp.choice("max_depth", [3, 4, 5, 6, 7, 8]),
        "learning_rate":  hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        "subsample":      hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "min_child_weight": hp.choice("min_child_weight", [1, 3, 5, 7]),
        "reg_alpha":      hp.loguniform("reg_alpha", np.log(0.001), np.log(10)),
        "reg_lambda":     hp.loguniform("reg_lambda", np.log(0.001), np.log(10)),
        "scale_pos_weight": hp.uniform("scale_pos_weight", 1, 10),
    }

    def objective(params):
        with mlflow.start_run(nested=True):
            model = train_xgboost(params, X_train, y_train, X_test, y_test)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            metrics = compute_metrics(y_test, y_pred, y_proba)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            return {"loss": -metrics["roc_auc"], "status": STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective, space=search_space, algo=tpe.suggest,
                max_evals=max_evals, trials=trials)
    logger.info(f"Best hyperparameters found: {best}")
    return best


def run_training_pipeline(
    spark,
    catalog: str = "prod_catalog",
    schema: str = "gold",
    table: str = "transactions_features",
    target_col: str = "is_fraud",
    max_evals: int = 30,
    register_model: bool = True,
):
    """Pipeline completo de treinamento MLOps com rastreamento MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_training_data(spark, catalog, schema, table)

    feature_cols = [c for c in df.columns if c not in [target_col, "_valid_from", "_valid_to", "_is_current", "_checksum"]]

    X_train, X_test, y_train, y_test, scaler = prepare_features(df, target_col, feature_cols)

    with mlflow.start_run(run_name="xgboost-hyperopt-search") as run:
        mlflow.set_tags({
            "model_type": "xgboost",
            "optimization": "hyperopt-bayesian",
            "data_source": f"{catalog}.{schema}.{table}",
            "feature_count": str(len(feature_cols)),
            "train_samples": str(len(X_train)),
        })

        best_params = hyperparameter_search(X_train, y_train, X_test, y_test, max_evals)

        logger.info("Treinando modelo final com os melhores hiperparâmetros...")
        final_model = train_xgboost(best_params, X_train, y_train, X_test, y_test)

        y_pred = final_model.predict(X_test)
        y_proba = final_model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"feature_importance": dict(zip(feature_cols, final_model.feature_importances_.tolist()))}, "feature_importance.json")

        signature = infer_signature(X_train, y_pred)
        model_info = mlflow.xgboost.log_model(
            xgb_model=final_model,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME if register_model else None,
        )

        logger.info(f"Model logged: {model_info.model_uri}")
        logger.info(f"Metrics: {metrics}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        return run.info.run_id, metrics


if __name__ == "__main__":
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    run_id, metrics = run_training_pipeline(spark)
    print(f"Training complete. Run ID: {run_id}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")
