"""Load the processed tables and define the time-based splits."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

FEATURES = [
    "h_points_last5", "h_gf5", "h_ga5", "h_gf10", "h_ga10", "h_rest",
    "a_points_last5", "a_gf5", "a_ga5", "a_gf10", "a_ga10", "a_rest",
]
LABELS = ["H", "D", "A"]
LAST_COMPLETED_SEASON = "2526"

# Matches up to and including this season can be used for the final refit.
# 2026/27 is still in progress and is held out for the demo.
ALL_TRAINING = f"Season <= '{LAST_COMPLETED_SEASON}'"
TRAIN = "Season <= '2324'"
VALID = "Season == '2425'"
TEST = "Season == '2526'"


def load_model_table(path: Path = None) -> pd.DataFrame:
    """Read model_table.csv, parse dates and cap summer rest days at 14."""
    path = path or DATA_DIR / "model_table.csv"
    df = pd.read_csv(path, parse_dates=["Date"], dtype={"Season": str})
    df["h_rest"] = df["h_rest"].clip(upper=14)
    df["a_rest"] = df["a_rest"].clip(upper=14)
    return df


def load_team_form(path: Path = None) -> pd.DataFrame:
    """Read the per-team rolling-form table from notebook 03."""
    path = path or DATA_DIR / "team_form.csv"
    return pd.read_csv(path, parse_dates=["Date"], dtype={"Season": str})


def split(df: pd.DataFrame) -> dict:
    """Return train, validation, test and held-out slices by season."""
    return {
        "train": df.query(TRAIN),
        "valid": df.query(VALID),
        "test": df.query(TEST),
        "live": df[df["Season"] > LAST_COMPLETED_SEASON],
    }
