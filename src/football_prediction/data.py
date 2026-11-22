"""Load the processed tables and define the time-based splits."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

FEATURES = [
    "h_points_last5", "h_gf5", "h_ga5", "h_gf10", "h_ga10", "h_rest",
    "a_points_last5", "a_gf5", "a_ga5", "a_gf10", "a_ga10", "a_rest",
    "h_elo", "a_elo", "elo_diff",
]
LABELS = ["H", "D", "A"]
LAST_COMPLETED_SEASON = "2526"

# Matches up to and including this season can be used for the final refit.
# 2026/27 is still in progress and is held out for the demo.
ALL_TRAINING = f"Season <= '{LAST_COMPLETED_SEASON}'"
TRAIN = "Season <= '2324'"
VALID = "Season == '2425'"
TEST = "Season == '2526'"


def compute_elo(matches: pd.DataFrame, k: float = 20.0,
                 home_adv: float = 60.0, init: float = 1500.0) -> pd.DataFrame:
    """Pre-match Elo ratings from match results, in chronological order.

    Ratings are updated after each match, so the returned values use only
    information available before kickoff. Promoted/new teams start at `init`.
    """
    m = matches.sort_values("Date").reset_index(drop=True)
    ratings = {}
    rows = []
    for _, r in m.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        rh = ratings.get(h, init)
        ra = ratings.get(a, init)
        rows.append((rh, ra, rh - ra))
        if r["FTR"] == "H":
            sh, sa = 1.0, 0.0
        elif r["FTR"] == "A":
            sh, sa = 0.0, 1.0
        else:
            sh, sa = 0.5, 0.5
        eh = 1.0 / (1.0 + 10.0 ** ((ra - (rh + home_adv)) / 400.0))
        gd = abs(int(r["FTHG"]) - int(r["FTAG"]))
        mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11.0 + gd) / 8.0
        ratings[h] = rh + k * mult * (sh - eh)
        ratings[a] = ra + k * mult * (sa - (1.0 - eh))
    return pd.DataFrame(rows, columns=["h_elo", "a_elo", "elo_diff"])


def load_model_table(path: Path = None, matches_path: Path = None) -> pd.DataFrame:
    """Read model_table.csv, cap rest days, and add pre-match Elo features."""
    path = path or DATA_DIR / "model_table.csv"
    matches_path = matches_path or DATA_DIR / "matches.csv"
    df = pd.read_csv(path, parse_dates=["Date"], dtype={"Season": str})
    df["h_rest"] = df["h_rest"].clip(upper=14)
    df["a_rest"] = df["a_rest"].clip(upper=14)
    if not {"h_elo", "a_elo", "elo_diff"}.issubset(df.columns):
        m = pd.read_csv(matches_path, parse_dates=["Date"], dtype={"Season": str})
        elo = pd.concat([m[["Date", "HomeTeam", "AwayTeam"]].reset_index(drop=True),
                         compute_elo(m)], axis=1)
        df = df.merge(elo, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    for col in ["h_elo", "a_elo", "elo_diff"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(1500.0 if col != "elo_diff" else 0.0)
    return df


def load_matches(path: Path = None) -> pd.DataFrame:
    """Read the cleaned matches table."""
    path = path or DATA_DIR / "matches.csv"
    return pd.read_csv(path, parse_dates=["Date"], dtype={"Season": str})


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
