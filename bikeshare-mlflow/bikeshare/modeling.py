"""Construcción y entrenamiento de modelos."""

from argparse import Namespace

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from bikeshare.preprocessing import build_preprocessor


def train_models(
    args: Namespace,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[Pipeline, np.ndarray, Pipeline, np.ndarray]:
    """Ajusta Random Forest y regresión lineal de referencia."""
    model = Pipeline(
        [
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    min_samples_leaf=args.min_samples_leaf,
                    max_features=args.max_features,
                    random_state=args.random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    baseline = Pipeline(
        [
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("regressor", LinearRegression()),
        ]
    )
    baseline.fit(X_train, y_train)
    return model, model.predict(X_test), baseline, baseline.predict(X_test)
