"""Build pre-match features and predict a single fixture."""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .data import FEATURES, compute_elo
from .models import predict_goals
from .scorelines import most_likely_score, result_probs, scoreline_grid, top_scores

RESULT_NAMES = {"H": "Home win", "D": "Draw", "A": "Away win"}


def _last_form_row(team: str, date, team_form: pd.DataFrame, venue: Optional[str]):
    """The team's form row for this match, or its latest row before kickoff."""
    rows = team_form[team_form["Team"] == team]
    if venue is not None:
        exact = rows[(rows["Date"] == date) & (rows["Venue"] == venue)]
        if not exact.empty:
            return exact.iloc[-1], True
    prior = rows[rows["Date"] <= date]
    if prior.empty:
        raise ValueError(f"No prior matches found for {team} before {date.date()}")
    return prior.iloc[-1], False


def _elo_as_of(home: str, away: str, date, matches: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Pre-match Elo ratings for a fixture as of `date`."""
    if matches is None:
        return {"h_elo": 1500.0, "a_elo": 1500.0, "elo_diff": 0.0}
    m = matches.sort_values("Date").reset_index(drop=True)
    elo = compute_elo(m)
    m = pd.concat([m[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]], elo], axis=1)
    exact = m[(m["HomeTeam"] == home) & (m["AwayTeam"] == away) & (m["Date"] == pd.Timestamp(date))]
    if not exact.empty:
        r = exact.iloc[-1]
        return {"h_elo": float(r["h_elo"]), "a_elo": float(r["a_elo"]), "elo_diff": float(r["elo_diff"])}
    # Future/unknown fixture: replay Elo up to date to get current ratings.
    before = m[m["Date"] < pd.Timestamp(date)]
    if before.empty:
        return {"h_elo": 1500.0, "a_elo": 1500.0, "elo_diff": 0.0}
    ratings = {}
    for _, r in before.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        rh, ra = float(r["h_elo"]), float(r["a_elo"])
        if r["FTR"] == "H":
            sh, sa = 1.0, 0.0
        elif r["FTR"] == "A":
            sh, sa = 0.0, 1.0
        else:
            sh, sa = 0.5, 0.5
        eh = 1.0 / (1.0 + 10.0 ** ((ra - (rh + 60.0)) / 400.0))
        gd = abs(int(r["FTHG"]) - int(r["FTAG"]))
        mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11.0 + gd) / 8.0
        ratings[h] = rh + 20.0 * mult * (sh - eh)
        ratings[a] = ra + 20.0 * mult * (sa - (1.0 - eh))
    rh = ratings.get(home, 1500.0)
    ra = ratings.get(away, 1500.0)
    return {"h_elo": rh, "a_elo": ra, "elo_diff": rh - ra}


def match_features(home: str, away: str, date, team_form: pd.DataFrame,
                   matches: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row of pre-match features for a home/away fixture.

    For a match already in the data the exact pre-match form row is used. For a
    future fixture the teams' most recent form is used and days of rest are
    measured from their last game to the requested date. Elo is computed from
    `matches` when provided.
    """
    date = pd.Timestamp(date)
    teams = set(team_form["Team"].unique())
    for name in (home, away):
        if name not in teams:
            raise ValueError(f"Unknown team {name!r}. Valid names: {sorted(teams)}")

    out = {}
    for prefix, team, venue in (("h_", home, "H"), ("a_", away, "A")):
        row, exact = _last_form_row(team, date, team_form, venue)
        if row["prior_matches"] < 5:
            raise ValueError(f"{team} has fewer than 5 matches of history before {date.date()}")
        out[prefix + "points_last5"] = row["points_last5"]
        out[prefix + "gf5"] = row["goals_for_last5"]
        out[prefix + "ga5"] = row["goals_against_last5"]
        out[prefix + "gf10"] = row["goals_for_last10"]
        out[prefix + "ga10"] = row["goals_against_last10"]
        rest = row["days_rest"] if exact else (date - row["Date"]).days
        out[prefix + "rest"] = np.nan if pd.isna(rest) else min(float(rest), 14.0)

    out.update(_elo_as_of(home, away, date, matches))
    return pd.DataFrame([out])[list(FEATURES)]


def predict_match(home: str, away: str, date, models: Dict,
                  team_form: pd.DataFrame, matches: Optional[pd.DataFrame] = None) -> Dict:
    """Expected goals, scoreline grid and WDL probabilities for one fixture."""
    X = match_features(home, away, date, team_form, matches)
    lam_h, lam_a = predict_goals(models, X)
    lam_h, lam_a = float(lam_h[0]), float(lam_a[0])
    p = result_probs([lam_h], [lam_a])[0]
    probs = dict(zip("HDA", p))
    predicted = max(probs, key=probs.get)
    ph, pa = most_likely_score(lam_h, lam_a, result=predicted)
    return {
        "home": home, "away": away, "date": pd.Timestamp(date),
        "lam_h": lam_h, "lam_a": lam_a,
        "grid": scoreline_grid([lam_h], [lam_a])[0],
        "probs": probs,
        "predicted": predicted,
        "score": (ph, pa),
        "top_scores": top_scores(lam_h, lam_a),
        "features": X,
    }
