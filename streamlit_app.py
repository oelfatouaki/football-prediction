"""Streamlit prototype: pick two teams and see the model's prediction.

Run with:  uv run streamlit run streamlit_app.py
The joblib artifact is created by notebook 07.
"""

import pandas as pd
import streamlit as st

from football_prediction import load_models, load_team_form, predict_match
from football_prediction.scorelines import MAX_GOALS

st.set_page_config(page_title="Premier League match predictor", page_icon="⚽",
                   layout="centered")


@st.cache_resource(show_spinner="Loading model and form data…")
def get_assets():
    return load_models(), load_team_form()


try:
    models, form = get_assets()
except FileNotFoundError:
    st.error("No saved model found. Run notebook `07_final_evaluation.ipynb` first "
             "to create `models/poisson.joblib`, then refresh this page.")
    st.stop()

teams = sorted(form["Team"].unique())
latest = form["Date"].max()

st.title("Premier League match predictor")
st.caption(
    "Poisson regression on recent form. Expected goals become a probability for "
    "every scoreline and for the home win / draw / away win outcomes. "
    f"Models refitted on all completed seasons through {latest.year - 1}/{str(latest.year)[-2:]}."
)

with st.sidebar:
    st.header("Fixture")
    home = st.selectbox("Home team", teams, index=teams.index("Man City") if "Man City" in teams else 0)
    away = st.selectbox("Away team", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)
    as_of = st.date_input("Kick-off date", value=latest.date(),
                          help="Form is taken from matches played before this date. "
                               "Dates after the last match reuse the latest available form.")
    same_team = home == away

if same_team:
    st.warning("Pick two different teams.")
    st.stop()

try:
    pred = predict_match(home, away, as_of, models, form)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.subheader(f"{home} vs {away}")
st.caption(f"Form as of {pred['date'].date()}")

c1, c2, c3 = st.columns(3)
c1.metric("Expected goals", f"{pred['lam_h']:.2f}", home)
c2.metric("Expected goals", f"{pred['lam_a']:.2f}", away)
c3.metric("Most likely score", f"{pred['score'][0]}-{pred['score'][1]}",
          pred["predicted"] + " win" if pred["predicted"] != "D" else "draw")

st.subheader("Win / draw / win chances")
names = {"H": f"{home} win", "D": "Draw", "A": f"{away} win"}
for key in "HDA":
    label = names[key]
    prob = pred["probs"][key]
    st.write(f"**{label}** — {prob:.1%}")
    st.progress(float(prob))
st.caption("Probabilities are derived from the scoreline grid and add up to 100%.")

st.subheader("Scoreline grid")
st.caption("Rows are home goals, columns are away goals. Darker cells are more likely.")
labels = list(range(MAX_GOALS + 1))
grid = pd.DataFrame(pred["grid"], index=labels, columns=labels)
grid.index.name = "Home \\ Away"
st.dataframe(
    grid.style.format("{:.1%}").background_gradient(cmap="Blues", axis=None),
    width="stretch",
)

st.subheader("Five most likely scorelines")
top = pd.DataFrame(
    [{"Score": f"{h}-{a}", "Probability": f"{p:.1%}"} for h, a, p in pred["top_scores"]]
)
st.table(top.set_index("Score"))

st.caption(
    "Model: independent Poisson goals, features from the last 5 and 10 matches only. "
    "Bookmaker odds, lineups and injuries are not used. See notebooks 06 and 07 for evaluation."
)
