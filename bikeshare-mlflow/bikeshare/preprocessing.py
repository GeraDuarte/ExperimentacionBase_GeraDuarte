"""Creación de variables, partición temporal y transformaciones."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bikeshare.settings import CATEGORICAL_FEATURES, FEATURE_COLUMNS, TARGET


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Deriva atributos de calendario y selecciona X e y."""
    model_data = data.copy()
    model_data["Month"] = model_data["Date"].dt.month
    model_data["DayOfWeek"] = model_data["Date"].dt.dayofweek
    model_data["IsWeekend"] = (model_data["DayOfWeek"] >= 5).astype(int)
    return model_data[FEATURE_COLUMNS], model_data[TARGET]


def split_chronologically(
    features: pd.DataFrame, target: pd.Series, test_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reserva el tramo final ordenado como prueba temporal."""
    split_index = int(len(features) * (1 - test_fraction))
    if split_index <= 0 or split_index >= len(features):
        raise ValueError("La fracción de prueba no produce train/test no vacíos.")
    return (
        features.iloc[:split_index],
        features.iloc[split_index:],
        target.iloc[:split_index],
        target.iloc[split_index:],
    )


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Construye imputación numérica y codificación categórica."""
    numeric_features = [
        column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES
    ]
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
