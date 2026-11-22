"""Scoring helpers shared by the evaluation notebooks and the app.

Log loss is computed explicitly rather than with sklearn's ``log_loss``: that
function sorts labels alphabetically (A, D, H), while our probability columns
are ordered (H, D, A), so calling it directly would silently swap home and away.
"""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

from .data import LABELS


def result_metrics(name: str, p, y_true) -> Dict[str, float]:
    """Accuracy and log loss for an (n, 3) array of H/D/A probabilities."""
    p = np.asarray(p, dtype=float)
    truth = pd.Series(y_true).map({"H": 0, "D": 1, "A": 2}).to_numpy()
    pred = np.array(LABELS)[p.argmax(axis=1)]
    p_of_truth = np.clip(p[np.arange(len(p)), truth], 1e-12, None)
    return {
        "model": name,
        "accuracy": float((pred == np.asarray(y_true)).mean()),
        "log loss": float(-np.log(p_of_truth).mean()),
    }


def goal_metrics(name: str, pred_home, pred_away, data) -> Dict[str, float]:
    """MAE and Poisson deviance for the two goal counts."""
    return {
        "model": name,
        "MAE home": mean_absolute_error(data["home_goals"], pred_home),
        "MAE away": mean_absolute_error(data["away_goals"], pred_away),
        "Poisson dev home": mean_poisson_deviance(data["home_goals"], pred_home),
        "Poisson dev away": mean_poisson_deviance(data["away_goals"], pred_away),
    }


def implied_probs(odds) -> np.ndarray:
    """Bookmaker odds to probabilities, with the margin removed."""
    raw = 1 / np.asarray(odds, dtype=float)
    return raw / raw.sum(axis=-1, keepdims=True)
