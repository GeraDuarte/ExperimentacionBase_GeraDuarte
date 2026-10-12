"""CLI para ejecutar experimentos de demanda de bicicletas."""

import json

from bikeshare.cli import parse_args
from bikeshare.pipeline import run_experiment
from bikeshare.settings import TRACKING_DB


def main() -> None:
    result = run_experiment(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        "MLflow UI: mlflow ui --backend-store-uri "
        f"sqlite:///{TRACKING_DB.as_posix()}"
    )


if __name__ == "__main__":
    main()
