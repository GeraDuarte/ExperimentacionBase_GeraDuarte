"""Carga del archivo de datos para el pipeline de bicicletas."""

from pathlib import Path

import pandas as pd

from bikeshare.settings import CATEGORICAL_FEATURES, NUMERIC_COLUMNS


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """Lee SeoulBikeData y comprueba que contenga las columnas necesarias."""
    if not data_path.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {data_path.resolve()}")

    data = pd.read_csv(data_path, encoding="cp1252")
    required_columns = {"Date", *NUMERIC_COLUMNS, *CATEGORICAL_FEATURES}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "El dataset no contiene las columnas requeridas: "
            + ", ".join(sorted(missing_columns))
        )
    return data
