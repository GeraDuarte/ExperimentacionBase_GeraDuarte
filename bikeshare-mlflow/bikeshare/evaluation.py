"""Métricas y artefactos gráficos de evaluación."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(
    actual: pd.Series, predictions: np.ndarray, prefix: str = ""
) -> dict[str, float]:
    return {
        f"{prefix}mae": float(mean_absolute_error(actual, predictions)),
        f"{prefix}rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        f"{prefix}r2": float(r2_score(actual, predictions)),
    }


def create_evaluation_artifacts(
    output_dir: Path,
    y_test: pd.Series,
    predictions: np.ndarray,
    metrics: dict[str, float],
    params: dict[str, object],
) -> list[Path]:
    """Guarda el resumen de métricas y el gráfico de residuos."""
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {"parameters": params, "metrics": metrics},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(y_test, predictions, alpha=0.35, s=14, color="#197278")
    bounds = [
        min(float(y_test.min()), float(predictions.min())),
        max(float(y_test.max()), float(predictions.max())),
    ]
    axes[0].plot(bounds, bounds, "--", color="#c44536")
    axes[0].set(
        title="Predicción vs. valor real",
        xlabel="Rentas reales",
        ylabel="Rentas predichas",
    )
    axes[1].hist(y_test.to_numpy() - predictions, bins=35, color="#386641")
    axes[1].axvline(0, color="#c44536", linestyle="--")
    axes[1].set(title="Residuos", xlabel="Real - predicho", ylabel="Frecuencia")
    figure.tight_layout()
    chart_path = output_dir / "evaluacion.png"
    figure.savefig(chart_path, dpi=140)
    plt.close(figure)
    return [metrics_path, chart_path]
