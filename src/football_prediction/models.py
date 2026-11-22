"""Fit, save and load the two Poisson goal models."""

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, MODEL_DIR


def fit_poisson(train: pd.DataFrame, target: str,
                features=FEATURES, alpha: float = 0.5) -> Pipeline:
    """Fit a standardised Poisson regression for one side's goals."""
    model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=2000))
    model.fit(train[list(features)], train[target])
    return model


def fit_models(train: pd.DataFrame, features=FEATURES,
               alpha: float = 0.5) -> Dict[str, Pipeline]:
    """Fit one model for home goals and one for away goals."""
    return {
        "home": fit_poisson(train, "home_goals", features, alpha),
        "away": fit_poisson(train, "away_goals", features, alpha),
    }


def predict_goals(models: Dict[str, Pipeline], X: pd.DataFrame,
                  features=FEATURES) -> Tuple[np.ndarray, np.ndarray]:
    """Return expected home and away goals for each row of X."""
    X = X[list(features)]
    return models["home"].predict(X), models["away"].predict(X)


def save_models(models: Dict[str, Pipeline], path: Path = None,
                features=FEATURES, trained_on: str = "") -> Path:
    """Save both pipelines and their metadata in one joblib artifact."""
    path = path or MODEL_DIR / "poisson.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "home": models["home"],
        "away": models["away"],
        "features": list(features),
        "trained_on": trained_on,
    }
    joblib.dump(artifact, path)
    return path


def load_models(path: Path = None) -> Dict[str, Pipeline]:
    """Load a joblib artifact written by :func:`save_models`."""
    path = path or MODEL_DIR / "poisson.joblib"
    return joblib.load(path)
