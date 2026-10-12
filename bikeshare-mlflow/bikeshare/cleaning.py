"""Limpieza, conversión de tipos y validación de observaciones."""

import pandas as pd

from bikeshare.settings import NUMERIC_COLUMNS, TARGET


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Quita duplicados y filas inválidas y entrega los datos cronológicamente."""
    cleaned = data.drop_duplicates().copy()
    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned["Date"] = pd.to_datetime(
        cleaned["Date"], format="%d/%m/%Y", errors="coerce"
    )

    invalid_rows = (
        cleaned["Date"].isna()
        | cleaned[TARGET].isna()
        | (cleaned[TARGET] < 0)
        | ~cleaned["Hour"].between(0, 23)
    )
    cleaned = cleaned.loc[~invalid_rows].copy()
    cleaned["Hour"] = cleaned["Hour"].astype(int)
    if cleaned.empty:
        raise ValueError("No quedan registros válidos después de limpiar el dataset.")
    return cleaned.sort_values(["Date", "Hour"]).reset_index(drop=True)
