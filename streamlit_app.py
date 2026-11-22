"""Professional Premier League match prediction app.

Run: uv run streamlit run streamlit_app.py
Requires models/poisson.joblib (create with notebook 07).
"""

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from football_prediction import load_matches, load_model_table, load_models, load_team_form, predict_match
from football_prediction.metrics import implied_probs
from football_prediction.scorelines import MAX_GOALS

st.set_page_config(
    page_title="PL Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading prediction model…")
def get_assets():
    return load_models(), load_team_form(), load_model_table(), load_matches()


try:
    models, form, matches, raw_matches = get_assets()
except FileNotFoundError:
    st.error(
        "**Model artifact not found.** Run `notebooks/07_final_evaluation.ipynb` to generate "
        "`models/poisson.joblib`, then refresh this page."
    )
    st.stop()

TEAMS = sorted(form["Team"].unique())
LATEST = pd.Timestamp(form["Date"].max()).date()
TRAINED_ON = models.get("trained_on", "available data")

TEAM_COLORS = {
    "Arsenal": "#EF0107", "Aston Villa": "#670E36", "Bournemouth": "#DA291C",
    "Brentford": "#E30613", "Brighton": "#0057B8", "Burnley": "#6C1D45",
    "Cardiff": "#0070B5", "Chelsea": "#034694", "Coventry": "#1D76C9",
    "Crystal Palace": "#1B458F", "Everton": "#003399", "Fulham": "#111111",
    "Huddersfield": "#0E63AD", "Hull": "#F18A01", "Ipswich": "#0033A0",
    "Leeds": "#1D428A", "Leicester": "#003090", "Liverpool": "#C8102E",
    "Luton": "#F78F1E", "Man City": "#6CABDD", "Man United": "#DA291C",
    "Newcastle": "#241F20", "Norwich": "#00A650", "Nott'm Forest": "#DD0000",
    "Sheffield United": "#EE2737", "Southampton": "#D71920", "Stoke": "#E03A3E",
    "Sunderland": "#EB172B", "Swansea": "#111111", "Tottenham": "#132257",
    "Watford": "#FBEE23", "West Brom": "#122F67", "West Ham": "#7A263A",
    "Wolves": "#FDB913",
}


def team_color(name: str) -> str:
    return TEAM_COLORS.get(name, "#334155")


def readable_text(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#0F172A" if luminance > 155 else "#FFFFFF"


def find_actual(home: str, away: str, kick_date: date):
    row = matches[
        (matches["HomeTeam"] == home)
        & (matches["AwayTeam"] == away)
        & (matches["Date"].dt.date == kick_date)
    ]
    return row.iloc[0] if not row.empty else None


def fmt_pct(value: float) -> str:
    return f"{value:.1%}"


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
  --bg: #F5F7FB;
  --surface: #FFFFFF;
  --surface-alt: #F8FAFC;
  --text: #0F172A;
  --muted: #64748B;
  --border: #E2E8F0;
  --shadow: 0 1px 3px rgba(15,23,42,0.04), 0 8px 24px rgba(15,23,42,0.04);
  --radius: 20px;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(180deg, #F8FAFC 0%, #EEF2F7 100%); }
.block-container { max-width: 1240px; padding-top: 1.25rem; padding-bottom: 2.5rem; }

.app-header {
  display:flex; justify-content:space-between; align-items:flex-end; gap:1rem;
  padding: 0.5rem 0 1.25rem;
}
.brand { font-size: 0.78rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; color: #475569; }
.app-header h1 { margin: .25rem 0 0; font-size: clamp(2rem, 4vw, 3rem); line-height: 1.05; letter-spacing: -0.04em; color: var(--text); font-weight: 850; }
.app-header p { margin: .6rem 0 0; color: var(--muted); font-size: 1rem; max-width: 720px; }
.header-meta { display:flex; gap:.5rem; flex-wrap:wrap; justify-content:flex-end; }
.meta-pill {
  background: rgba(255,255,255,.8); border: 1px solid var(--border); color: #334155;
  padding: .45rem .75rem; border-radius: 999px; font-size: .76rem; font-weight: 600; backdrop-filter: blur(8px);
}

.panel {
  background: var(--surface); border: 1px solid rgba(226,232,240,.8); border-radius: var(--radius);
  box-shadow: var(--shadow);
}
.panel-body { padding: 1.4rem; }
.section-label {
  font-size: .74rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; color: #64748B; margin-bottom: .85rem;
}

.input-wrap .stSelectbox label, .input-wrap .stDateInput label {
  font-size: .8rem !important; font-weight: 650 !important; color: #334155 !important;
}
.input-wrap div[data-baseweb="select"] > div, .input-wrap div[data-baseweb="input"] > div {
  border-radius: 12px; border-color: #DCE5EF; background: #FBFDFF;
}

.verdict-card {
  border-radius: 24px; padding: 1.5rem; color: white; position: relative; overflow: hidden;
  min-height: 220px; display:flex; flex-direction:column; justify-content:space-between;
  box-shadow: 0 12px 30px rgba(15,23,42,.12);
}
.verdict-card::after {
  content:''; position:absolute; right:-60px; top:-60px; width:180px; height:180px;
  background: rgba(255,255,255,.12); border-radius: 50%;
}
.verdict-kicker { font-size:.74rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; opacity:.88; }
.verdict-winner { font-size: clamp(1.8rem, 3vw, 2.8rem); font-weight: 850; letter-spacing: -.04em; margin: .35rem 0 .2rem; line-height:1.05; }
.verdict-sub { font-size: .98rem; opacity:.94; }
.verdict-score {
  display:inline-flex; gap:.5rem; align-items:center; margin-top:1rem;
  padding: .55rem .9rem; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.14);
  border-radius: 999px; font-weight: 700; font-size:.9rem; backdrop-filter: blur(4px);
}

.stat-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap: .9rem; }
.stat {
  background: var(--surface-alt); border: 1px solid var(--border); border-radius: 16px; padding: 1rem; text-align:center;
}
.stat-value { font-size: 1.75rem; font-weight: 800; letter-spacing: -.03em; color: var(--text); line-height:1; }
.stat-label { font-size: .74rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing:.08em; margin-top:.45rem; }
.stat-sub { font-size: .8rem; color: #475569; margin-top:.3rem; }

.prob-item { margin-bottom: 1rem; }
.prob-item:last-child { margin-bottom: 0; }
.prob-top { display:flex; justify-content:space-between; align-items:baseline; gap:.75rem; margin-bottom:.45rem; }
.prob-name { font-size:.93rem; font-weight:650; color:#1E293B; }
.prob-value { font-size:.93rem; font-weight:800; color:var(--text); }
.prob-book { font-size:.78rem; color:var(--muted); font-weight:600; margin-left:.35rem; }
.prob-track { width:100%; height:12px; background:#E8EEF5; border-radius:999px; overflow:hidden; }
.prob-fill { height:100%; border-radius:999px; transition: width .4s ease; }

.score-cards { display:grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap:.7rem; }
.score-card {
  background: var(--surface-alt); border: 1px solid var(--border); border-radius: 14px; padding: .85rem .5rem; text-align:center;
}
.score-card .score { font-size:1.25rem; font-weight:800; color:var(--text); letter-spacing:-.02em; }
.score-card .pct { font-size:.78rem; color:var(--muted); font-weight:600; margin-top:.2rem; }

.insight {
  display:flex; gap:.75rem; align-items:flex-start; padding: .9rem 1rem; border-radius: 14px;
  background: #F8FAFC; border: 1px solid var(--border); color:#334155; font-size:.88rem;
}
.insight strong { color: var(--text); }

.actual-card {
  background: linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 100%); border:1px solid var(--border);
  border-radius: 16px; padding: 1rem 1.1rem;
}
.actual-score { font-size:1.5rem; font-weight:800; letter-spacing:-.03em; color:var(--text); }
.actual-meta { color:var(--muted); font-size:.86rem; margin-top:.25rem; }

div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 14px; overflow:hidden; }
.stButton > button {
  width:100%; border-radius: 12px; height: 2.9rem; font-weight: 700; border: none;
  background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); color: white;
}
.stButton > button:hover { background: linear-gradient(135deg, #1E293B 0%, #334155 100%); color: white; }

.footer-note { text-align:center; color: var(--muted); font-size: .8rem; margin-top: 1.5rem; padding: 1rem 0 .25rem; }
@media (max-width: 900px) {
  .score-cards { grid-template-columns: repeat(3, minmax(0,1fr)); }
  .stat-grid { grid-template-columns: 1fr; }
  .app-header { flex-direction: column; align-items: flex-start; }
  .header-meta { justify-content: flex-start; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    f"""
<div class="app-header">
  <div>
    <div class="brand">Football Analytics</div>
    <h1>Match Prediction</h1>
    <p>Expected goals, scoreline probabilities, and win/draw/win forecasts powered by Poisson regression and pre-match Elo.</p>
  </div>
  <div class="header-meta">
    <div class="meta-pill">Poisson + Elo</div>
    <div class="meta-pill">Trained through {TRAINED_ON}</div>
    <div class="meta-pill">Latest data: {LATEST:%d %b %Y}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

with st.container():
    st.markdown('<div class="panel"><div class="panel-body input-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Select fixture</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1.25, 1.25, 1, 0.9], gap="medium")
    with c1:
        home = st.selectbox("Home team", TEAMS, index=TEAMS.index("Man City") if "Man City" in TEAMS else 0)
    with c2:
        away_options = [t for t in TEAMS if t != home]
        default_away = "Chelsea" if "Chelsea" in away_options else away_options[0]
        away = st.selectbox("Away team", away_options, index=away_options.index(default_away))
    with c3:
        kick_date = st.date_input(
            "Kick-off date",
            value=LATEST,
            min_value=date(2017, 8, 1),
            max_value=date(2030, 12, 31),
        )
    with c4:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        run = st.button("Predict match", use_container_width=True)
    st.markdown(
        "<div style='margin-top:.75rem;color:#64748B;font-size:.82rem;'>"
        "Tip: choose a future date to use the latest available form and current-season Elo. "
        "Historical fixtures in the dataset will also show the actual result and bookmaker comparison."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

try:
    pred = predict_match(home, away, kick_date, models, form, raw_matches)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

probs = pred["probs"]
predicted = pred["predicted"]
score_h, score_a = pred["score"]
home_c = team_color(home)
away_c = team_color(away)

if predicted == "H":
    verdict_team = home
    verdict_prob = probs["H"]
    verdict_color = home_c
    verdict_text = readable_text(home_c)
elif predicted == "A":
    verdict_team = away
    verdict_prob = probs["A"]
    verdict_color = away_c
    verdict_text = readable_text(away_c)
else:
    verdict_team = "Draw"
    verdict_prob = probs["D"]
    verdict_color = "#334155"
    verdict_text = "#FFFFFF"

actual = find_actual(home, away, kick_date)
book = None
if actual is not None and not actual[["B365H", "B365D", "B365A"]].isna().any():
    b = implied_probs(actual[["B365H", "B365D", "B365A"]].to_numpy(dtype=float))
    book = dict(zip("HDA", b))

# Confidence label
if verdict_prob >= 0.55:
    confidence = "High confidence"
elif verdict_prob >= 0.45:
    confidence = "Moderate confidence"
else:
    confidence = "Close matchup"

st.write("")
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown(
        f"""
<div class="verdict-card" style="background:linear-gradient(135deg, {verdict_color} 0%, {verdict_color}E8 58%, #0F172A 155%); color:{verdict_text};">
  <div>
    <div class="verdict-kicker">Likely outcome · {confidence}</div>
    <div class="verdict-winner">{verdict_team}</div>
    <div class="verdict-sub">Win probability <strong>{fmt_pct(verdict_prob)}</strong> · projected score <strong>{score_h}–{score_a}</strong></div>
  </div>
  <div class="verdict-score">{home} {score_h}–{score_a} {away}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown('<div class="panel"><div class="panel-body">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Match expectations</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="stat-grid">
  <div class="stat">
    <div class="stat-value">{pred['lam_h']:.2f}</div>
    <div class="stat-label">Home xG</div>
    <div class="stat-sub">{home}</div>
  </div>
  <div class="stat">
    <div class="stat-value">{pred['lam_a']:.2f}</div>
    <div class="stat-label">Away xG</div>
    <div class="stat-sub">{away}</div>
  </div>
  <div class="stat">
    <div class="stat-value">{score_h}–{score_a}</div>
    <div class="stat-label">Projected score</div>
    <div class="stat-sub">Most likely in predicted result</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="panel"><div class="panel-body">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Outcome probabilities</div>', unsafe_allow_html=True)
    for key, label, color in [
        ("H", f"{home} win", home_c),
        ("D", "Draw", "#64748B"),
        ("A", f"{away} win", away_c),
    ]:
        p = probs[key]
        book_html = f"<span class='prob-book'>Book {fmt_pct(book[key])}</span>" if book else ""
        st.markdown(
            f"""
<div class="prob-item">
  <div class="prob-top">
    <div class="prob-name">{label}</div>
    <div class="prob-value">{fmt_pct(p)}{book_html}</div>
  </div>
  <div class="prob-track"><div class="prob-fill" style="width:{p*100:.2f}%;background:{color};"></div></div>
</div>
""",
            unsafe_allow_html=True,
        )
    if book:
        st.markdown(
            "<div style='margin-top:.85rem;color:#64748B;font-size:.8rem;'>Bookmaker values are Bet365 implied probabilities with margin removed.</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div></div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel"><div class="panel-body">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Most likely scorelines</div>', unsafe_allow_html=True)
    score_cards = "".join(
        f"""<div class="score-card"><div class="score">{h}–{a}</div><div class="pct">{fmt_pct(p)}</div></div>"""
        for h, a, p in pred["top_scores"]
    )
    st.markdown(f'<div class="score-cards">{score_cards}</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='margin-top:.9rem;color:#64748B;font-size:.82rem;'>Exact scores are low-probability events; use these as plausible scenarios.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="panel"><div class="panel-body">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Prediction insight</div>', unsafe_allow_html=True)
    elo_diff = pred["features"]["elo_diff"].iloc[0]
    fav = home if elo_diff >= 0 else away
    st.markdown(
        f"""
<div class="insight">
  <div>
    <strong>{fav}</strong> has the higher pre-match Elo by <strong>{abs(elo_diff):.0f} points</strong>.
    The model combines this team-strength signal with recent goals, points, and rest days to estimate expected goals.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="insight" style="margin-top:.7rem;">
  <div>
    Form data is taken from matches played before <strong>{kick_date:%d %b %Y}</strong>. 
    For future fixtures, the latest available team form and current-season Elo are used.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div></div>", unsafe_allow_html=True)

    if actual is not None:
        st.write("")
        hg, ag = int(actual["home_goals"]), int(actual["away_goals"])
        actual_res = actual["FTR"]
        actual_name = {"H": f"{home} win", "D": "Draw", "A": f"{away} win"}[actual_res]
        hit = "Matched" if predicted == actual_res else "Different"
        st.markdown('<div class="panel"><div class="panel-body">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Actual result</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="actual-card">
  <div class="actual-score">{home} {hg}–{ag} {away}</div>
  <div class="actual-meta">{actual_name} · Outcome prediction: <strong>{hit}</strong> · 
  Model gave this result {fmt_pct(probs[actual_res])} and this exact score {fmt_pct(pred['grid'][hg, ag])}.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("</div></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Scoreline grid
# ---------------------------------------------------------------------------

st.write("")
st.markdown('<div class="panel"><div class="panel-body">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Scoreline probability grid</div>', unsafe_allow_html=True)
g1, g2 = st.columns([1, 2])
with g1:
    max_show = st.slider("Grid size", 5, MAX_GOALS + 1, 7)
with g2:
    st.markdown(
        "<div style='color:#64748B;font-size:.84rem;padding-top:2rem;'>Rows are home goals, columns are away goals. "
        "Each cell shows the model probability for that exact scoreline.</div>",
        unsafe_allow_html=True,
    )

grid = pred["grid"][:max_show, :max_show]
labels = list(range(max_show))
grid_df = pd.DataFrame(grid, index=labels, columns=labels)
grid_df.index.name = f"{home} goals \\ {away} goals"
styled = (
    grid_df.style.format("{:.1%}")
    .background_gradient(cmap="Blues", axis=None, vmin=0, vmax=float(grid.max()))
    .set_properties(**{"text-align": "center", "font-size": "0.82rem"})
)
st.dataframe(styled, width="stretch", height=min(520, 45 + 38 * max_show))
st.markdown("</div></div>", unsafe_allow_html=True)

with st.expander("Model details"):
    st.markdown(
        """
**Model**: Two independent Poisson regressors (home goals, away goals).  
**Features**: recent form (points, goals for/against over last 5 and 10 matches), rest days, and pre-match Elo (`h_elo`, `a_elo`, `elo_diff`).  
**Training data**: Premier League matches through the latest available completed/played seasons in `data/processed`.  
**Limitations**: No lineups, injuries, transfers, tactical changes, or xG. Exact scorelines are inherently uncertain.

Bookmaker probabilities, when shown, are for comparison only and are not model inputs.
"""
    )

st.markdown(
    "<div class='footer-note'>Premier League Match Predictor · Poisson + Elo model · "
    "For analysis and educational use, not betting advice.</div>",
    unsafe_allow_html=True,
)
