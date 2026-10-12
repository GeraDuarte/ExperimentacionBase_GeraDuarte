"""Ejecuta veinte corridas de sensibilidad, variando un hiperparámetro a la vez."""

import subprocess
import sys
from pathlib import Path

from bikeshare.settings import EXPERIMENT_NAME


SCRIPT = Path(__file__).resolve().parent / "run_experiment.py"
BASE_PARAMETERS = {
    "n_estimators": "120",
    "max_depth": "12",
    "min_samples_leaf": "1",
    "max_features": "1.0",
}
SWEEPS = {
    "n_estimators": ("40", "80", "120", "160", "240"),
    "max_depth": ("4", "8", "12", "20", "30"),
    "min_samples_leaf": ("1", "2", "4", "8", "16"),
    "max_features": ("0.2", "0.4", "0.6", "0.8", "1.0"),
}
CLI_NAMES = {
    "n_estimators": "--n-estimators",
    "max_depth": "--max-depth",
    "min_samples_leaf": "--min-samples-leaf",
    "max_features": "--max-features",
}


def main() -> None:
    expected_runs = sum(len(values) for values in SWEEPS.values())
    completed_runs = 0
    for parameter, values in SWEEPS.items():
        for value in values:
            parameters = {**BASE_PARAMETERS, parameter: value}
            command = [
                sys.executable,
                str(SCRIPT),
                "--experiment-name",
                EXPERIMENT_NAME,
                "--sweep-parameter",
                parameter,
                "--sweep-value",
                value,
                "--random-state",
                "42",
            ]
            for name, parameter_value in parameters.items():
                command.extend([CLI_NAMES[name], parameter_value])

            completed_runs += 1
            print(
                f"[{completed_runs}/{expected_runs}] "
                f"{parameter}={value}; parámetros restantes fijos",
                flush=True,
            )
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
