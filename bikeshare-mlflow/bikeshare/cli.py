"""Argumentos de línea de comandos y validación de parámetros."""

import argparse
from pathlib import Path

from bikeshare.settings import DEFAULT_DATA_PATH, EXPERIMENT_NAME


def positive_int(value: str) -> int:
    """Valida hiperparámetros que deben ser enteros positivos."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("debe ser un entero mayor que cero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("debe ser un entero mayor o igual que cero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena Random Forest y registra métricas/modelo en MLflow."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--n-estimators", type=positive_int, default=120)
    parser.add_argument("--max-depth", type=positive_int, default=12)
    parser.add_argument("--min-samples-leaf", type=positive_int, default=1)
    parser.add_argument("--max-features", type=float, default=1.0)
    parser.add_argument(
        "--random-state", "--seed", dest="random_state", type=nonnegative_int, default=42
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--sweep-parameter")
    parser.add_argument("--sweep-value")
    args = parser.parse_args()
    if not 0 < args.max_features <= 1:
        parser.error("--max-features debe estar en el intervalo (0, 1].")
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction debe estar en el intervalo (0, 1).")
    if (args.sweep_parameter is None) != (args.sweep_value is None):
        parser.error("--sweep-parameter y --sweep-value deben usarse juntos.")
    return args
