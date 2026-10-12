"""Configuración de MLflow y registro de cada corrida."""

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from bikeshare.settings import ARTIFACT_DIR, CATEGORICAL_FEATURES, FEATURE_COLUMNS, TRACKING_DB


def configure_experiment(name: str) -> str:
    """Selecciona el backend SQLite y crea o recupera el experimento."""
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        return client.create_experiment(name, artifact_location=ARTIFACT_DIR.as_uri())
    return experiment.experiment_id


def log_run(
    experiment_id: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[Path],
    model: Pipeline,
    input_example: pd.DataFrame,
) -> str:
    """Registra parámetros, métricas, artefactos y modelo."""
    numeric_features = [
        column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES
    ]
    model_input = input_example.copy()
    model_input[numeric_features] = model_input[numeric_features].astype(float)
    signature = infer_signature(model_input, model.predict(model_input))

    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run(run_name=run_name) as run:
        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.set_tags(
            {
                "dataset": "SeoulBikeData",
                "problem_type": "regression",
                "evaluation": "temporal_holdout",
                "baseline": "LinearRegression",
            }
        )
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact), artifact_path="evaluation")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest",
            signature=signature,
            input_example=model_input,
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )
        return run.info.run_id
