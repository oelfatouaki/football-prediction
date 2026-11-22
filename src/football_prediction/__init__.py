"""Predict Premier League goals, scorelines and results."""

from .data import FEATURES, LABELS, load_model_table, load_team_form, split
from .metrics import goal_metrics, implied_probs, result_metrics
from .models import fit_models, load_models, predict_goals, save_models
from .predict import match_features, predict_match
from .scorelines import most_likely_score, result_probs, scoreline_grid, top_scores

__all__ = [
    "FEATURES", "LABELS", "load_model_table", "load_team_form", "split",
    "result_metrics", "goal_metrics", "implied_probs",
    "fit_models", "save_models", "load_models", "predict_goals",
    "match_features", "predict_match",
    "scoreline_grid", "result_probs", "most_likely_score", "top_scores",
    "main",
]


def main() -> None:
    print("Run `streamlit run streamlit_app.py` for the prediction demo, "
          "or open the notebooks in order (01 to 07).")
