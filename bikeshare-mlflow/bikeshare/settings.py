"""Constantes comunes para el pipeline de entrenamiento."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "SeoulBikeData.csv"
TRACKING_DB = WORKSPACE_DIR / "mlflow.db"
ARTIFACT_DIR = WORKSPACE_DIR / "mlruns"
EXPERIMENT_NAME = "SeoulBikeData - sensibilidad hiperparametros (20 corridas)"
TARGET = "Rented Bike Count"
CATEGORICAL_FEATURES = ["Seasons", "Holiday", "Functioning Day"]
NUMERIC_COLUMNS = [
    TARGET,
    "Hour",
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
]
FEATURE_COLUMNS = [
    "Hour",
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
    "Month",
    "DayOfWeek",
    "IsWeekend",
    *CATEGORICAL_FEATURES,
]
