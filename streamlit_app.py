"""Professional Streamlit prototype for Premier League match predictions.

Run: uv run streamlit run streamlit_app.py
Requires: models/poisson.joblib (create with notebook 07).
"""

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from football_prediction import load_model_table, load_models, load_team_form, predict_match
from football_prediction.metrics import implied_probs
from football_prediction.scorelines import MAX_GOALS

st.set_page_config(page_title="PL Match Predictor", page_icon="⚽", layout="wide")

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def get_assets():
    models = load_models()
    form = load_team_form()
    matches = load_model_table()
    return models, form, matches


try:
    models, form, matches = get_assets()
except FileNotFoundError:
    st.error(
        "Saved model not found. Run `notebooks/07_final_evaluation.ipynb` to create "
        "`models/poisson.joblib`, then refresh."
    )
    st.stop()

TEAMS = sorted(form["Team"].unique())
LATEST = pd.Timestamp(form["Date"].max()).date()
TRAINED_ON = models.get("trained_on", "completed seasons")

# Team accent colors (primary-ish; used sparingly)
TEAM_COLORS = {
    "Arsenal": "#EF0107", "Aston Villa": "#670E36", "Bournemouth": "#DA291C",
    "Brentford": "#D20000", "Brighton": "#0057B8", "Burnley": "#6C1D45",
    "Cardiff": "#0070B5", "Chelsea": "#034694", "Coventry": "#77BBFF",
    "Crystal Palace": "#1B458F", "Everton": "#003399", "Fulham": "#000000",
    "Huddersfield": "#0E63AD", "Hull": "#F18A01", "Ipswich": "#0033A0",
    "Leeds": "#FFCD00", "Leicester": "#003090", "Liverpool": "#C8102E",
    "Luton": "#F78F1E", "Man City": "#6CABDD", "Man United": "#DA291C",
    "Newcastle": "#241F20", "Norwich": "#00A650", "Nott'm Forest": "#DD0000",
    "Sheffield United": "#EE2737", "Southampton": "#D71920", "Stoke": "#E03A3E",
    "Sunderland": "#EB172B", "Swansea": "#121212", "Tottenham": "#132257",
    "Watford": "#FBEE23", "West Brom": "#122F67", "West Ham": "#7A263A",
    "Wolves": "#FDB913",
}

def team_color(name: str) -> str:
    return TEAM_COLORS.get(name, "#4A5568")


def text_color(bg_hex: str) -> str:
    """Pick white/black for readable text on a team color."""
    h = bg_hex.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#111827" if lum > 160 else "#FFFFFF"


def find_actual(home: str, away: str, kick_date: date):
    row = matches[
        (matches["HomeTeam"] == home)
        & (matches["AwayTeam"] == away)
        & (matches["Date"].dt.date == kick_date)
    ]
    return row.iloc[0] if not row.empty else None


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1280px; }
.hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0b3b2e 100%);
  color: #f8fafc; padding: 28px 32px; border-radius: 18px; margin-bottom: 1.25rem;
  border: 1px solid rgba(255,255,255,0.06);
}
.hero h1 { margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; }
.hero p { margin: 0.45rem 0 0; color: #cbd5e1; font-size: 0.98rem; max-width: 820px; }
.pill {
  display:inline-block; padding: 0.28rem 0.7rem; border-radius: 999px;
  background: rgba(255,255,255,0.08); color:#e2e8f0; font-size:0.75rem; font-weight:600;
  margin-right:0.4rem; margin-top:0.8rem; border:1px solid rgba(255,255,255,0.08);
}
.card {
  background: #ffffff; border: 1px solid #e5e7eb; border-radius: 16px;
  padding: 1.2rem 1.25rem; box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}
.card-title { font-size:0.78rem; font-weight:700; color:#64748b; text-transform:uppercase;
  letter-spacing:.08em; margin-bottom:0.5rem; }
.verdict {
  border-radius: 18px; padding: 1.4rem; color: white; min-height: 168px;
  display:flex; flex-direction:column; justify-content:space-between;
}
.verdict-label { font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; opacity:.9; }
.verdict-team { font-size:2rem; font-weight:800; line-height:1.1; margin:0.35rem 0; letter-spacing:-0.03em; }
.verdict-sub { font-size:0.95rem; opacity:.92; }
.metric-box { text-align:center; padding:0.9rem 0.5rem; }
.metric-value { font-size:2rem; font-weight:800; color:#0f172a; letter-spacing:-0.03em; }
.metric-label { font-size:0.78rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }
.metric-sub { font-size:0.83rem; color:#475569; margin-top:0.15rem; }
.prob-row { margin-bottom: 0.85rem; }
.prob-head { display:flex; justify-content:space-between; font-size:0.92rem; margin-bottom:0.35rem; }
.prob-name { font-weight:600; color:#1e293b; }
.prob-val { font-weight:700; color:#0f172a; }
.bar-bg { background:#e2e8f0; border-radius:999px; height:12px; overflow:hidden; }
.bar-fill { height:12px; border-radius:999px; }
.score-pill {
  display:inline-flex; align-items:center; gap:0.6rem; background:#f8fafc; border:1px solid #e2e8f0;
  border-radius:12px; padding:0.55rem 0.8rem; margin:0 0.45rem 0.45rem 0; font-size:0.9rem;
}
.score-pill strong { font-size:1rem; color:#0f172a; }
.muted { color:#64748b; font-size:0.88rem; }
.small-note { color:#64748b; font-size:0.8rem; line-height:1.5; }
div[data-testid="stSidebar"] { background:#f8fafc; border-right:1px solid #e5e7eb; }
div[data-testid="stSidebar"] .block-container { padding-top:1.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    f"""
<div class="hero">
  <h1>Premier League Match Predictor</h1>
  <p>Pick a fixture and a kick-off date. The model estimates expected goals from recent form,
  then turns them into scoreline probabilities and a win / draw / win forecast.</p>
  <span class="pill">Poisson regression</span>
  <span class="pill">Trained through {TRAINED_ON}</span>
  <span class="pill">Test log loss 1.046 vs bookmaker 1.016</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Match setup")
    home = st.selectbox("Home team", TEAMS, index=TEAMS.index("Man City") if "Man City" in TEAMS else 0)
    away_options = [t for t in TEAMS if t != home]
    away_default = "Chelsea" if "Chelsea" in away_options else away_options[0]
    away = st.selectbox("Away team", away_options, index=away_options.index(away_default))
    kick_date = st.date_input(
        "Kick-off date",
        value=LATEST,
        min_value=date(2017, 8, 1),
        max_value=date(2030, 12, 31),
        help="Uses each team's form before this date. Future dates use the latest available form.",
    )
    predict_btn = st.button("Generate prediction", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown(
        f"<div class='small-note'><b>Data coverage</b><br>Latest match in data: {LATEST}. "
        f"The in-progress 2026/27 season is held out from training.<br><br>"
        f"<b>Model</b><br>Two Poisson regressions (home/away goals) on last-5/last-10 form and rest days. "
        f"No lineups, injuries, xG, or bookmaker odds used as features.</div>",
        unsafe_allow_html=True,
    )

# Use session state so first load shows a prediction
if "ran" not in st.session_state:
    st.session_state.ran = True
if predict_btn:
    st.session_state.ran = True

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

try:
    pred = predict_match(home, away, kick_date, models, form)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

probs = pred["probs"]
predicted = pred["predicted"]
score_h, score_a = pred["score"]
home_c = team_color(home)
away_c = team_color(away)
if predicted == "H":
    verdict_color, verdict_text, verdict_team = home_c, text_color(home_c), home
    verdict_prob = probs["H"]
elif predicted == "A":
    verdict_color, verdict_text, verdict_team = away_c, text_color(away_c), away
    verdict_prob = probs["A"]
else:
    verdict_color, verdict_text, verdict_team = "#334155", "#FFFFFF", "Draw"
    verdict_prob = probs["D"]

actual = find_actual(home, away, kick_date)
book = None
if actual is not None and not actual[["B365H", "B365D", "B365A"]].isna().any():
    b = implied_probs(actual[["B365H", "B365D", "B365A"]].to_numpy(dtype=float))
    book = dict(zip("HDA", b))

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Fixture</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns([1, 0.45, 1])
    with m1:
        st.markdown(
            f"<div class='metric-box'><div class='metric-label'>Home</div>"
            f"<div class='metric-value' style='font-size:1.55rem;color:{home_c}'>{home}</div></div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            "<div class='metric-box'><div class='metric-label'>Date</div>"
            f"<div class='metric-sub' style='font-size:1rem;font-weight:700;color:#0f172a'>{kick_date:%d %b %Y}</div>"
            "<div class='metric-sub'>vs</div></div>",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"<div class='metric-box'><div class='metric-label'>Away</div>"
            f"<div class='metric-value' style='font-size:1.55rem;color:{away_c}'>{away}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin:0.4rem 0 1rem;border:none;border-top:1px solid #e5e7eb'>",
                unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(
            f"<div class='metric-box'><div class='metric-value'>{pred['lam_h']:.2f}</div>"
            f"<div class='metric-label'>Expected home goals</div></div>",
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            f"<div class='metric-box'><div class='metric-value'>{pred['lam_a']:.2f}</div>"
            f"<div class='metric-label'>Expected away goals</div></div>",
            unsafe_allow_html=True,
        )
    with g3:
        st.markdown(
            f"<div class='metric-box'><div class='metric-value'>{score_h}-{score_a}</div>"
            f"<div class='metric-label'>Most likely score</div>"
            f"<div class='metric-sub'>within predicted result</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Win / draw / win probability</div>', unsafe_allow_html=True)

    for key, label, color in [
        ("H", f"{home} win", home_c),
        ("D", "Draw", "#64748b"),
        ("A", f"{away} win", away_c),
    ]:
        p = probs[key]
        book_txt = f" &nbsp;<span style='color:#64748b;font-weight:500'>Book {book[key]:.1%}</span>" if book else ""
        st.markdown(
            f"""
<div class="prob-row">
  <div class="prob-head"><span class="prob-name">{label}</span>
  <span class="prob-val">{p:.1%}{book_txt}</span></div>
  <div class="bar-bg"><div class="bar-fill" style="width:{p*100:.2f}%;background:{color}"></div></div>
</div>
""",
            unsafe_allow_html=True,
        )
    if book:
        st.markdown("<div class='small-note'>“Book” shows Bet365 implied probability with the margin removed, available for historical matches in the dataset.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown(
        f"""
<div class="verdict" style="background:linear-gradient(135deg,{verdict_color} 0%, {verdict_color}dd 60%, #0f172a 140%); color:{verdict_text};">
  <div>
    <div class="verdict-label">Model verdict</div>
    <div class="verdict-team">{verdict_team}</div>
    <div class="verdict-sub">Most likely outcome: <b>{verdict_prob:.1%}</b> probability</div>
  </div>
  <div class="verdict-sub" style="opacity:.85">Projected scoreline: <b>{home} {score_h}–{score_a} {away}</b></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Top scorelines</div>', unsafe_allow_html=True)
    pills = []
    for h, a, p in pred["top_scores"]:
        pills.append(f"<span class='score-pill'><strong>{h}–{a}</strong> <span class='muted'>{p:.1%}</span></span>")
    st.markdown("".join(pills), unsafe_allow_html=True)
    st.markdown("<div class='small-note'>Individual exact scores are naturally low-probability; use them as likely scenarios, not certainties.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if actual is not None:
        st.write("")
        hg, ag = int(actual["home_goals"]), int(actual["away_goals"])
        actual_res = actual["FTR"]
        actual_name = {"H": f"{home} win", "D": "Draw", "A": f"{away} win"}[actual_res]
        hit = "correct" if predicted == actual_res else "different"
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Actual result</div>', unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:1.35rem;font-weight:800;color:#0f172a'>{home} {hg}–{ag} {away}</div>"
            f"<div class='muted' style='margin-top:0.25rem'>{actual_name} · model outcome was {hit} "
            f"· model gave this result {probs[actual_res]:.1%} and this exact score "
            f"{pred['grid'][hg, ag]:.1%}.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Scoreline grid
# ---------------------------------------------------------------------------

st.write("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Scoreline probability grid</div>', unsafe_allow_html=True)
st.markdown("<div class='small-note'>Rows: home goals. Columns: away goals. Each cell is P(home goals = row, away goals = column).</div>", unsafe_allow_html=True)

max_show = st.slider("Grid size", min_value=6, max_value=MAX_GOALS + 1, value=8, help="Show scores from 0 up to this value.")
grid = pred["grid"][:max_show, :max_show]
labels = list(range(max_show))
grid_df = pd.DataFrame(grid, index=labels, columns=labels)
grid_df.index.name = f"{home} \\ {away}"

styled = (
    grid_df.style
    .format("{:.1%}")
    .background_gradient(cmap="YlGnBu", axis=None, vmin=0, vmax=float(grid.max()))
    .set_properties(**{"font-size": "0.82rem", "text-align": "center"})
)
st.dataframe(styled, width="stretch", height=min(560, 42 + 35 * max_show))
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer / explanation
# ---------------------------------------------------------------------------

with st.expander("How to read this prediction"):
    st.markdown(
        """
- **Expected goals** are the model's average goals for each team in similar recent-form situations, not a guaranteed score.
- **Win/draw/win** probabilities come from summing all scorelines in the 0–10 grid. They add to 100%.
- **Most likely score** is the highest-probability scoreline *inside the predicted outcome* (home win / draw / away win), so it agrees with the verdict.
- **Top scorelines** are the highest-probability exact scores overall. 1-1 is often near the top because football scores cluster low.
- **Bookmaker comparison** appears only when the selected fixture/date exists in the historical dataset with Bet365 odds.

This is a form-based Poisson model. It does not know lineups, injuries, transfers, motivation, or red cards.
"""
    )

st.markdown(
    "<div class='small-note' style='text-align:center;margin-top:1.2rem'>"
    "Built with scikit-learn Poisson regression · Evaluated on untouched 2025/26 season · "
    "Bookmaker baseline: Bet365 odds with margin removed"
    "</div>",
    unsafe_allow_html=True,
)
