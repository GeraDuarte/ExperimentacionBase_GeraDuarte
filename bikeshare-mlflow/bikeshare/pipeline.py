"""Orquestación del flujo de entrenamiento y registro."""

import random
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

from bikeshare.cleaning import clean_data
from bikeshare.data_loading import load_raw_data
from bikeshare.evaluation import calculate_metrics, create_evaluation_artifacts
from bikeshare.modeling import train_models
from bikeshare.preprocessing import prepare_features, split_chronologically
from bikeshare.tracking import configure_experiment, log_run


def run_experiment(args: Namespace) -> dict[str, Any]:
    """Ejecuta carga, limpieza, entrenamiento, evaluación y logging."""
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    raw_data = load_raw_data(args.data)
    data = clean_data(raw_data)
    features, target = prepare_features(data)
    X_train, X_test, y_train, y_test = split_chronologically(
        features, target, args.test_fraction
    )

    model, predictions, _, baseline_predictions = train_models(
        args, X_train, y_train, X_test
    )
    metrics = calculate_metrics(y_test, predictions)
    metrics.update(
        calculate_metrics(y_test, baseline_predictions, prefix="baseline_")
    )

    params = {
        "model": "RandomForestRegressor",
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "random_state": args.random_state,
        "test_fraction": args.test_fraction,
        "split_strategy": "chronological",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    if args.sweep_parameter is not None:
        params["sweep_parameter"] = args.sweep_parameter
        params["sweep_value"] = args.sweep_value

    experiment_id = configure_experiment(args.experiment_name)
    with tempfile.TemporaryDirectory(prefix="bike-demand-") as temp_dir:
        artifacts = create_evaluation_artifacts(
            Path(temp_dir), y_test, predictions, metrics, params
        )
        run_id = log_run(
            experiment_id=experiment_id,
            run_name=(
                f"sweep-{args.sweep_parameter}-{args.sweep_value}"
                if args.sweep_parameter is not None
                else f"rf-depth-{args.max_depth}-trees-{args.n_estimators}"
            ),
            params=params,
            metrics=metrics,
            artifacts=artifacts,
            model=model,
            input_example=X_test.head(5),
        )

    return {
        "run_id": run_id,
        "experiment": args.experiment_name,
        "metrics": metrics,
        "test_period": (
            str(data["Date"].iloc[len(X_train)].date()),
            str(data["Date"].iloc[-1].date()),
        ),
    }
