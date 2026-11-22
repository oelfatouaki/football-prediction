from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / 'reports' / 'figures'
OUT = ROOT / 'reports' / 'Football_Prediction_Project_Report.pdf'

PAGE_W, PAGE_H = 8.27, 11.69  # A4 inches
MARGIN_X = 0.75
TOP = PAGE_H - 0.75
BOTTOM = 0.75
CONTENT_W = PAGE_W - 2 * MARGIN_X


def new_page():
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.axis('off')
    return fig, ax


def draw_title(ax, title, subtitle=None):
    ax.text(MARGIN_X, TOP, title, fontsize=28, fontweight='bold', color='#0f172a', va='top')
    y = TOP - 0.55
    if subtitle:
        ax.text(MARGIN_X, y, subtitle, fontsize=12, color='#475569', va='top')
        y -= 0.35
    ax.plot([MARGIN_X, PAGE_W - MARGIN_X], [y, y], color='#cbd5e1', linewidth=1)
    return y - 0.35


def wrap_text(text, width=95):
    lines = []
    for para in text.split('\n'):
        para = para.rstrip()
        if not para:
            lines.append('')
        else:
            lines.extend(textwrap.wrap(para, width=width) or [''])
    return lines


def draw_paragraph(ax, y, text, size=10.5, color='#1e293b', leading=0.22, width=95, indent=0):
    for line in wrap_text(text, width=width):
        if y < BOTTOM:
            return y, True
        ax.text(MARGIN_X + indent, y, line, fontsize=size, color=color, va='top', family='DejaVu Sans')
        y -= leading if line else leading * 0.7
    return y, False


def draw_heading(ax, y, text, size=16, color='#0f172a'):
    if y < BOTTOM + 0.4:
        return y, True
    ax.text(MARGIN_X, y, text, fontsize=size, fontweight='bold', color=color, va='top')
    return y - 0.38, False


def draw_subheading(ax, y, text, size=12.5, color='#0f172a'):
    if y < BOTTOM + 0.3:
        return y, True
    ax.text(MARGIN_X, y, text, fontsize=size, fontweight='bold', color=color, va='top')
    return y - 0.3, False


def draw_bullets(ax, y, items, size=10.5, width=92):
    for item in items:
        lines = wrap_text(item, width=width)
        for i, line in enumerate(lines):
            if y < BOTTOM:
                return y, True
            prefix = '• ' if i == 0 else '  '
            ax.text(MARGIN_X + 0.15, y, prefix + line, fontsize=size, color='#1e293b', va='top')
            y -= 0.22
        y -= 0.03
    return y, False


def draw_image(ax, y, image_path, max_width=CONTENT_W, max_height=3.2, caption=None):
    if not Path(image_path).exists():
        return draw_paragraph(ax, y, f'[Missing image: {image_path}]')
    img = plt.imread(image_path)
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    dw, dh = w * scale, h * scale
    if y - dh - (0.3 if caption else 0) < BOTTOM:
        return y, True
    x = MARGIN_X + (CONTENT_W - dw) / 2
    ax.imshow(img, extent=[x, x + dw, y - dh, y], aspect='auto')
    y -= dh + 0.1
    if caption:
        ax.text(PAGE_W / 2, y, caption, fontsize=9, color='#475569', ha='center', va='top', style='italic')
        y -= 0.28
    return y, False


def draw_table(ax, y, df, col_widths=None, fontsize=9, row_h=0.28):
    if df.empty:
        return y, False
    cols = list(df.columns)
    ncols = len(cols)
    if col_widths is None:
        col_widths = [CONTENT_W / ncols] * ncols
    total_w = sum(col_widths)
    x0 = MARGIN_X
    # header
    if y - row_h < BOTTOM:
        return y, True
    ax.add_patch(plt.Rectangle((x0, y - row_h), total_w, row_h, facecolor='#0f172a', edgecolor='#0f172a'))
    x = x0
    for c, w in zip(cols, col_widths):
        ax.text(x + 0.08, y - row_h / 2, str(c), fontsize=fontsize, fontweight='bold', color='white', va='center')
        x += w
    y -= row_h
    for i, (_, row) in enumerate(df.iterrows()):
        if y - row_h < BOTTOM:
            return y, True
        bg = '#f8fafc' if i % 2 == 0 else '#ffffff'
        ax.add_patch(plt.Rectangle((x0, y - row_h), total_w, row_h, facecolor=bg, edgecolor='#e2e8f0', linewidth=0.5))
        x = x0
        for c, w in zip(cols, col_widths):
            val = row[c]
            txt = f'{val:.3f}' if isinstance(val, float) else str(val)
            ax.text(x + 0.08, y - row_h / 2, txt, fontsize=fontsize, color='#1e293b', va='center')
            x += w
        y -= row_h
    return y - 0.15, False


def add_cover(pdf):
    fig, ax = new_page()
    ax.add_patch(plt.Rectangle((0, 0), PAGE_W, PAGE_H, facecolor='#0f172a'))
    ax.add_patch(plt.Rectangle((0, 0), PAGE_W, 2.2, facecolor='#111827', alpha=0.25))
    ax.text(MARGIN_X, 8.4, 'PREMIER LEAGUE', fontsize=13, fontweight='bold', color='#93c5fd')
    ax.text(MARGIN_X, 7.8, 'Football Match\nPrediction Project', fontsize=38, fontweight='bold', color='white', va='top', linespacing=1.15)
    ax.text(MARGIN_X, 6.2, 'A detailed end-to-end report covering data collection, cleaning, SQL feature engineering,\nPoisson + Elo modeling, evaluation, and the Streamlit prediction app.',
            fontsize=12, color='#cbd5e1', va='top', linespacing=1.5)
    info = [
        ('Author', 'Oussama El Fatouaki'),
        ('Repository', 'github.com/oelfatouaki/football-prediction'),
        ('Date', 'November 22, 2026'),
        ('Scope', 'Premier League 2017/18–2026/27 (played matches)'),
    ]
    y = 4.2
    for k, v in info:
        ax.text(MARGIN_X, y, k.upper(), fontsize=8.5, fontweight='bold', color='#94a3b8')
        ax.text(MARGIN_X, y - 0.25, v, fontsize=11, color='white')
        y -= 0.75
    ax.text(MARGIN_X, 1.1, 'Built with Python, pandas, DuckDB, scikit-learn, SciPy, joblib, and Streamlit.', fontsize=9.5, color='#94a3b8')
    pdf.savefig(fig)
    plt.close(fig)


def section_pages(pdf, title, subtitle, blocks):
    """blocks: list of dicts with type heading/subheading/paragraph/bullets/image/table/spacer/pagebreak"""
    fig, ax = new_page()
    y = draw_title(ax, title, subtitle)
    for b in blocks:
        t = b['type']
        if t == 'pagebreak':
            pdf.savefig(fig); plt.close(fig)
            fig, ax = new_page(); y = TOP
            continue
        overflow = False
        if t == 'heading':
            y, overflow = draw_heading(ax, y, b['text'], size=b.get('size', 16))
        elif t == 'subheading':
            y, overflow = draw_subheading(ax, y, b['text'])
        elif t == 'paragraph':
            y, overflow = draw_paragraph(ax, y, b['text'], size=b.get('size', 10.5), width=b.get('width', 95))
        elif t == 'bullets':
            y, overflow = draw_bullets(ax, y, b['items'])
        elif t == 'image':
            y, overflow = draw_image(ax, y, b['path'], max_width=b.get('max_width', CONTENT_W), max_height=b.get('max_height', 3.2), caption=b.get('caption'))
        elif t == 'table':
            y, overflow = draw_table(ax, y, b['df'], col_widths=b.get('col_widths'))
        elif t == 'spacer':
            y -= b.get('h', 0.2)
        if overflow:
            pdf.savefig(fig); plt.close(fig)
            fig, ax = new_page(); y = TOP
            # retry same block on new page
            if t == 'heading':
                y, _ = draw_heading(ax, y, b['text'], size=b.get('size', 16))
            elif t == 'subheading':
                y, _ = draw_subheading(ax, y, b['text'])
            elif t == 'paragraph':
                y, _ = draw_paragraph(ax, y, b['text'], size=b.get('size', 10.5), width=b.get('width', 95))
            elif t == 'bullets':
                y, _ = draw_bullets(ax, y, b['items'])
            elif t == 'image':
                y, _ = draw_image(ax, y, b['path'], max_width=b.get('max_width', CONTENT_W), max_height=b.get('max_height', 3.2), caption=b.get('caption'))
            elif t == 'table':
                y, _ = draw_table(ax, y, b['df'], col_widths=b.get('col_widths'))
    pdf.savefig(fig)
    plt.close(fig)


def make_generated_figures():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = FIG_DIR / 'report'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split timeline
    fig, ax = plt.subplots(figsize=(9, 2.2))
    spans = [
        ('Train', 2017.6, 2024.5, '#2563eb'),
        ('Validation', 2024.5, 2025.5, '#f59e0b'),
        ('Test', 2025.5, 2026.5, '#10b981'),
        ('App refit includes\nplayed 2026/27', 2026.5, 2027.2, '#8b5cf6'),
    ]
    for label, s, e, c in spans:
        ax.barh(0, e - s, left=s, height=0.5, color=c, alpha=0.9)
        ax.text((s + e) / 2, 0, label, ha='center', va='center', color='white', fontsize=9, fontweight='bold')
    ax.set_xlim(2017.2, 2027.4)
    ax.set_yticks([])
    ax.set_xticks([2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027])
    ax.set_xlabel('Season start year', fontsize=9)
    ax.set_title('Time-based split (no random shuffling)', fontsize=11, fontweight='bold')
    ax.spines[['left', 'right', 'top']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / 'split_timeline.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    # Model comparison
    models = ['Constant', 'Baseline\ngoals->Poisson', 'Form-only\nPoisson', 'Poisson+Elo', 'Bookmaker']
    val_ll = [1.078, 1.078, 1.019, 1.007, 0.971]
    test_ll = [1.086, 1.081, 1.046, 1.041, 1.016]
    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(x - w/2, val_ll, w, label='Validation 2024/25', color='#3b82f6')
    ax.bar(x + w/2, test_ll, w, label='Test 2025/26', color='#10b981')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.5)
    ax.set_ylabel('Log loss (lower is better)')
    ax.set_title('Model comparison: validation and untouched test season', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0.9, 1.12)
    for i, v in enumerate(val_ll):
        ax.text(i - w/2, v + 0.003, f'{v:.3f}', ha='center', fontsize=7)
    for i, v in enumerate(test_ll):
        ax.text(i + w/2, v + 0.003, f'{v:.3f}', ha='center', fontsize=7)
    ax.spines[['right', 'top']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / 'model_comparison.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    # Pipeline diagram
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.axis('off')
    boxes = [
        ('Raw CSVs\nfootball-data.co.uk', '#e2e8f0', '#0f172a'),
        ('Cleaning\n01_data_cleaning', '#dbeafe', '#1d4ed8'),
        ('EDA + SQL\n02_eda, 03_sql', '#dcfce7', '#166534'),
        ('Training table\n04_training_table', '#fef3c7', '#92400e'),
        ('Poisson + Elo\n05-07', '#ede9fe', '#5b21b6'),
        ('Streamlit app\nprediction UI', '#fee2e2', '#991b1b'),
    ]
    for i, (label, fc, ec) in enumerate(boxes):
        x = 0.03 + i * 0.162
        rect = plt.Rectangle((x, 0.35), 0.135, 0.32, facecolor=fc, edgecolor=ec, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.0675, 0.51, label, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0f172a')
        if i < len(boxes) - 1:
            ax.annotate('', xy=(x + 0.162, 0.51), xytext=(x + 0.138, 0.51), arrowprops=dict(arrowstyle='->', lw=1.5, color='#475569'))
    ax.set_title('End-to-end project pipeline', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'pipeline.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    return out_dir


def build():
    out_dir = make_generated_figures()

    # key numbers from data
    matches = pd.read_csv(ROOT / 'data/processed/matches.csv', dtype={'Season': str})
    model_table = pd.read_csv(ROOT / 'data/processed/model_table.csv', dtype={'Season': str})

    with PdfPages(OUT) as pdf:
        add_cover(pdf)

        section_pages(pdf, '1. Project Overview', 'What the project does and what it delivers', [
            {'type': 'paragraph', 'text': 'This project predicts Premier League football matches end to end. For a selected fixture, it estimates how many goals each team is likely to score, converts those expected goals into probabilities for every plausible scoreline, and aggregates them into home win, draw, and away win probabilities. The model is evaluated against simple baselines and Bet365 bookmaker odds, then served through a Streamlit app.'},
            {'type': 'subheading', 'text': 'Main goals'},
            {'type': 'bullets', 'items': [
                'Build a clean, reproducible dataset from raw football-data.co.uk CSVs.',
                'Create pre-match features with no data leakage, using SQL window functions.',
                'Train interpretable Poisson regression models for home and away goals.',
                'Convert expected goals into scoreline and match-result probabilities.',
                'Evaluate honestly with a time-based split and an untouched test season.',
                'Improve predictions by adding pre-match Elo team-strength features.',
                'Deploy the final model in a professional Streamlit prediction app.',
            ]},
            {'type': 'subheading', 'text': 'End-to-end pipeline'},
            {'type': 'image', 'path': out_dir / 'pipeline.png', 'max_height': 2.1, 'caption': 'Project flow from raw data to the prediction app.'},
            {'type': 'subheading', 'text': 'Key deliverables'},
            {'type': 'bullets', 'items': [
                f'Cleaned matches dataset: {len(matches):,} rows, {matches["Season"].nunique()} season files (including in-progress 2026/27).',
                f'Modeling table: {len(model_table):,} rows after excluding matches with thin history (<5 prior matches).',
                'Notebooks 01–07 covering cleaning, EDA, SQL, training table, models, evaluation, and final export.',
                'Reusable Python package in src/football_prediction for data loading, features, metrics, prediction, and scorelines.',
                'Streamlit app (streamlit_app.py) for interactive fixture prediction.',
            ]},
        ])

        section_pages(pdf, '2. Data Source and Raw Data', 'football-data.co.uk Premier League CSVs', [
            {'type': 'paragraph', 'text': 'The raw data comes from football-data.co.uk Premier League match CSV files. Each season file contains match results and match statistics. The project uses nine completed seasons (2017/18 to 2025/26) and played matches from the in-progress 2026/27 season. Raw files are not tracked in git; cleaned processed files are tracked so the project is reproducible.'},
            {'type': 'subheading', 'text': 'Columns used'},
            {'type': 'bullets', 'items': [
                'Identifiers and result: Season, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR.',
                'Match stats (EDA only, never used as model features): shots, shots on target, fouls, corners, cards.',
                'Bookmaker odds for evaluation only: B365H, B365D, B365A.',
            ]},
            {'type': 'subheading', 'text': 'Data quality checks performed'},
            {'type': 'bullets', 'items': [
                'Date parsing and chronological ordering.',
                'Missing-value inspection and column standardization across seasons.',
                'Validation of team names and season coverage.',
                'Sanity checks: goals non-negative, result agrees with goals, odds positive.',
                'In notebook 03, rebuilt league table and confirmed goals-for equals goals-against.',
            ]},
            {'type': 'subheading', 'text': 'Why this data is suitable'},
            {'type': 'paragraph', 'text': 'Football-data.co.uk provides consistent match-level results across many seasons, including bookmaker odds for benchmarking. This makes it possible to build a baseline, a statistical model, and a market comparison using the same match rows.'},
        ])

        section_pages(pdf, '3. Data Cleaning (Notebook 01)', 'From raw season CSVs to matches.csv', [
            {'type': 'paragraph', 'text': 'Notebook 01_data_cleaning.ipynb reads raw season files, keeps the required columns, normalizes types, and combines them into a single clean dataset. The result is saved as data/processed/matches.csv.'},
            {'type': 'subheading', 'text': 'Cleaning steps'},
            {'type': 'bullets', 'items': [
                'Load each season CSV and append a Season code (e.g., 1718, 2425, 2627).',
                'Keep only relevant columns and rename where necessary for consistency.',
                'Parse Date into datetime; sort by date for downstream rolling features.',
                'Ensure numeric columns (goals, shots, odds) have correct types.',
                'Drop or handle incomplete rows where required fields are missing.',
                'Export cleaned data to data/processed/matches.csv.',
            ]},
            {'type': 'subheading', 'text': 'Output'},
            {'type': 'paragraph', 'text': f'The cleaned file contains {len(matches):,} matches with columns: {", ".join(matches.columns)}. This file is the foundation for all later notebooks and the app.'},
        ])

        section_pages(pdf, '4. Exploratory Data Analysis (Notebook 02)', 'Understanding patterns before modeling', [
            {'type': 'paragraph', 'text': 'Notebook 02_eda.ipynb explores the cleaned match data to understand scoring patterns, home advantage, and baseline predictability. EDA also motivates the model choice: goals are low-count events and home advantage is real, so Poisson goal modeling is a reasonable baseline.'},
            {'type': 'subheading', 'text': 'Main questions explored'},
            {'type': 'bullets', 'items': [
                'What is the distribution of match results (home win/draw/away win)?',
                'How strong is home advantage, and has it changed over seasons?',
                'How many goals are scored per match and per team?',
                'Which teams have strong home records?',
                'Do shots relate to goals, and why raw match shots cannot be features?',
            ]},
            {'type': 'image', 'path': FIG_DIR / 'result_shares.png', 'max_height': 2.6, 'caption': 'Distribution of match outcomes across the dataset.'},
            {'type': 'image', 'path': FIG_DIR / 'home_win_rate_by_season.png', 'max_height': 2.6, 'caption': 'Home win rate by season, showing home advantage varies over time.'},
        ])

        section_pages(pdf, '4. Exploratory Data Analysis (continued)', 'More EDA visuals', [
            {'type': 'image', 'path': FIG_DIR / 'goals_per_game.png', 'max_height': 2.7, 'caption': 'Goals per game patterns used to motivate Poisson modeling.'},
            {'type': 'image', 'path': FIG_DIR / 'shots_vs_goals.png', 'max_height': 2.7, 'caption': 'Shots relate to goals, but same-match shots are not used as pre-match features.'},
            {'type': 'image', 'path': FIG_DIR / 'top_home_records.png', 'max_height': 2.7, 'caption': 'Top home records provide context for team strength and home advantage.'},
        ])

        section_pages(pdf, '5. SQL Feature Engineering (Notebook 03)', 'DuckDB rolling features with no leakage', [
            {'type': 'paragraph', 'text': 'Notebook 03_sql.ipynb uses DuckDB SQL to transform match rows into team-match rows and compute rolling pre-match features. This is a key step: every feature must be available before kickoff. The notebook also validates the transformation by reconstructing a league table and checking goal totals.'},
            {'type': 'subheading', 'text': 'team_matches transformation'},
            {'type': 'bullets', 'items': [
                'Convert each match into two rows: one for the home team, one for the away team.',
                'For each team-match row, store goals for/against, venue, opponent, and points (3/1/0).',
                'Use UNION ALL over home and away perspectives.',
            ]},
            {'type': 'subheading', 'text': 'Rolling features in team_form'},
            {'type': 'bullets', 'items': [
                'points_last5: sum of points in previous 5 matches.',
                'goals_for_last5 / goals_against_last5: rolling average goals scored/conceded over 5.',
                'goals_for_last10 / goals_against_last10: rolling average over 10 for longer-term form.',
                'days_rest: date difference since previous match, capped later to avoid summer-break outliers.',
                'All rolling windows use ROWS BETWEEN N PRECEDING AND 1 PRECEDING to exclude the current match.',
            ]},
            {'type': 'subheading', 'text': 'Leakage prevention rule'},
            {'type': 'paragraph', 'text': 'The current match is never included in its own features. This is enforced by SQL window boundaries and by excluding same-match stats like shots, cards, fouls, and odds from model inputs. Leakage would make evaluation unrealistically good and fail in real prediction.'},
        ])

        section_pages(pdf, '6. Training Table and Time Split (Notebook 04)', 'One row per match, ready for modeling', [
            {'type': 'paragraph', 'text': 'Notebook 04_training_table.ipynb joins home and away team_form rows back to each match. The result is model_table.csv: one row per match with home-side and away-side pre-match features, targets, and odds for evaluation.'},
            {'type': 'subheading', 'text': 'Model table columns'},
            {'type': 'bullets', 'items': [
                'Targets: home_goals, away_goals (and FTR for result evaluation).',
                'Form features: h/a points_last5, gf5, ga5, gf10, ga10, rest.',
                'Identifiers/context: Season, Date, HomeTeam, AwayTeam.',
                'Evaluation-only odds: B365H, B365D, B365A.',
            ]},
            {'type': 'subheading', 'text': 'Time-based split'},
            {'type': 'image', 'path': out_dir / 'split_timeline.png', 'max_height': 1.9, 'caption': 'Training/validation/test split is chronological to simulate real forecasting.'},
            {'type': 'bullets', 'items': [
                'Train: seasons up to 2023/24.',
                'Validation: 2024/25, used to compare models and choose configuration.',
                'Test: 2025/26, touched once for final reporting.',
                'App artifact: refit on all available played matches, including 2026/27 played fixtures.',
            ]},
        ])

        section_pages(pdf, '7. Modeling (Notebooks 05, 07, src)', 'Poisson regression + Elo', [
            {'type': 'paragraph', 'text': 'The core model predicts expected goals for each team using two Poisson regressors: one for home goals and one for away goals. Poisson regression is appropriate because goals are non-negative counts and low-scoring events. The improved model adds pre-match Elo ratings as team-strength features.'},
            {'type': 'subheading', 'text': 'Features'},
            {'type': 'bullets', 'items': [
                'Recent form: h/a_points_last5, h/a_gf5, h/a_ga5, h/a_gf10, h/a_ga10, h/a_rest.',
                'Team strength: h_elo, a_elo, elo_diff (pre-match, chronologically updated).',
                'No same-match stats, no odds, no future information.',
            ]},
            {'type': 'subheading', 'text': 'Elo calculation'},
            {'type': 'paragraph', 'text': 'Elo ratings start at 1500 for unseen teams. Before each match, current ratings are stored as features. After the match, ratings are updated based on actual result versus expected result, with a home-advantage term and a goal-difference multiplier. Because updates happen after storing pre-match features, Elo is leakage-safe.'},
            {'type': 'subheading', 'text': 'Why two models?'},
            {'type': 'paragraph', 'text': 'Home and away scoring are modeled separately because home advantage and team roles differ. Each model receives the same match feature row and outputs an expected goal count (lambda).'},
        ])

        section_pages(pdf, '8. Scorelines and Probabilities (Notebook 06)', 'From expected goals to match odds', [
            {'type': 'paragraph', 'text': 'Notebook 06 converts expected goals into probabilities. For each possible scoreline, it assumes independent Poisson distributions for home and away goals: P(home=h, away=a) = Poisson(h|lambda_h) * Poisson(a|lambda_a). It computes a grid from 0-0 up to 10-10.'},
            {'type': 'subheading', 'text': 'From grid to result probabilities'},
            {'type': 'bullets', 'items': [
                'Home win probability: sum of grid cells where home goals > away goals.',
                'Draw probability: sum of diagonal cells where home goals == away goals.',
                'Away win probability: sum of cells where home goals < away goals.',
                'Probabilities are normalized by grid mass; the 0–10 grid covers almost all probability.',
            ]},
            {'type': 'subheading', 'text': 'Evaluation metrics'},
            {'type': 'bullets', 'items': [
                'Log loss: primary metric. Penalizes confident wrong probabilities; lower is better.',
                'Accuracy: whether the most likely outcome matches the actual result. Useful but noisy.',
                'MAE / Poisson deviance for goal predictions.',
                'Brier-style intuition is related, but project uses log loss for probability quality.',
            ]},
            {'type': 'paragraph', 'text': 'Important implementation detail: result_metrics computes log loss manually over H/D/A columns. sklearn.log_loss can sort labels alphabetically (A,D,H), which would silently swap probabilities if passed incorrectly.'},
        ])

        results_df = pd.DataFrame([
            ['Constant (training shares)', '41.1%', '1.078', '42.4%', '1.086'],
            ['Baseline goals -> Poisson', '41.1%', '1.078', '42.4%', '1.081'],
            ['Form-only Poisson', '49.6%', '1.019', '48.8%', '1.046'],
            ['Poisson + Elo', '51.7%', '1.007', '49.1%', '1.041'],
            ['Bookmaker (Bet365)', '54.1%', '0.971', '49.3%', '1.016'],
        ], columns=['Model', 'Val Acc', 'Val LogLoss', 'Test Acc', 'Test LogLoss'])

        section_pages(pdf, '9. Results and Evaluation', 'Validation, untouched test, and interpretation', [
            {'type': 'paragraph', 'text': 'Models were compared on 2024/25 validation (375 matches) and scored once on 2025/26 test (375 matches). All model training for these reported scores used data up to 2023/24 only. The bookmaker baseline uses Bet365 implied probabilities with margin removed.'},
            {'type': 'table', 'df': results_df, 'col_widths': [2.8, 0.9, 1.0, 0.9, 1.0]},
            {'type': 'image', 'path': out_dir / 'model_comparison.png', 'max_height': 2.8, 'caption': 'Log loss comparison. Lower is better.'},
            {'type': 'subheading', 'text': 'Interpretation'},
            {'type': 'bullets', 'items': [
                'Poisson + Elo improves over form-only on both validation and test log loss.',
                'The model closes most of the gap to bookmakers but does not fully beat them.',
                'Accuracy is close to bookmaker on test, but log loss still favors bookmakers, indicating better probability calibration/sharpness.',
                'Exact score prediction is intentionally low-confidence; probability distributions are the main output.',
            ]},
        ])

        section_pages(pdf, '10. Streamlit Application', 'Professional interactive prediction interface', [
            {'type': 'paragraph', 'text': 'The Streamlit app (streamlit_app.py) loads the saved model artifact and processed data. Users select home team, away team, and kick-off date. The app computes pre-match features using the latest available form and Elo before the selected date, then displays predictions.'},
            {'type': 'subheading', 'text': 'App features'},
            {'type': 'bullets', 'items': [
                'Fixture selection: home team, away team, date picker, predict action.',
                'Verdict card: likely outcome, confidence, and projected score.',
                'Expected goals: home and away lambda values.',
                'Win/draw/win probabilities with visual bars and optional bookmaker comparison for historical matches.',
                'Top 5 most likely scorelines and full scoreline probability grid heatmap.',
                'Actual result panel when selected fixture exists in historical data.',
                'Model details and limitations expander.',
            ]},
            {'type': 'subheading', 'text': 'Current-season behavior'},
            {'type': 'paragraph', 'text': 'The app artifact is refitted on available played matches through 2026/27 (played through 2026-11-08), so predictions can use current-season form and Elo. The reported 2025/26 test evaluation remains separate and honest because that evaluation model was trained only up to 2023/24.'},
            {'type': 'subheading', 'text': 'Run command'},
            {'type': 'paragraph', 'text': 'uv run streamlit run streamlit_app.py'},
        ])

        section_pages(pdf, '11. Repository Structure and Reproducibility', 'How to run the project', [
            {'type': 'subheading', 'text': 'Important files'},
            {'type': 'bullets', 'items': [
                'data/processed/: matches.csv, team_form.csv, model_table.csv.',
                'notebooks/01-07: cleaning, EDA, SQL, training table, models, scorelines/evaluation, final evaluation.',
                'src/football_prediction/: data.py, models.py, scorelines.py, metrics.py, predict.py.',
                'streamlit_app.py: interactive prediction app.',
                'models/poisson.joblib: saved model artifact (gitignored; regenerate).',
            ]},
            {'type': 'subheading', 'text': 'Reproduction commands'},
            {'type': 'bullets', 'items': [
                'git clone https://github.com/oelfatouaki/football-prediction',
                'cd football-prediction',
                'uv sync',
                'uv run jupyter lab  # run notebooks 01-07 as needed',
                'uv run streamlit run streamlit_app.py',
            ]},
            {'type': 'subheading', 'text': 'Reproducibility notes'},
            {'type': 'bullets', 'items': [
                'Raw CSVs are not tracked; processed data is tracked.',
                'Model artifact is gitignored; create with notebook 07.',
                'Time-based evaluation avoids random split leakage.',
                'All feature engineering uses only pre-match information.',
            ]},
        ])

        section_pages(pdf, '12. Limitations and Future Work', 'Honest assessment and next improvements', [
            {'type': 'subheading', 'text': 'Limitations'},
            {'type': 'bullets', 'items': [
                'Independent Poisson assumption ignores correlation between teams’ scoring in some match states.',
                'No lineup, injuries, transfers, motivation, weather, or tactical data.',
                'Elo captures team strength but not short-term squad availability.',
                'Bookmaker odds still outperform on log loss; market includes richer information.',
                'Current season is incomplete; future fixture predictions depend on latest available form.',
            ]},
            {'type': 'subheading', 'text': 'Future improvements'},
            {'type': 'bullets', 'items': [
                'Dixon-Coles correction for low-score dependence (0-0, 1-0, 0-1, 1-1).',
                'Gradient boosting / LightGBM Poisson objective with careful validation.',
                'xG-based team strength and attacking/defensive Elo.',
                'Calibration plots and probability reliability monitoring.',
                'Player availability/injury features and richer schedule/rest context.',
                'Backtesting service and automated report generation.',
            ]},
            {'type': 'subheading', 'text': 'Conclusion'},
            {'type': 'paragraph', 'text': 'The project delivers a complete, reproducible football prediction workflow: clean data, leakage-safe features, interpretable probabilistic modeling, honest evaluation, and a usable app. Poisson + Elo provides a strong baseline that is substantially better than naive baselines and reasonably close to bookmaker probabilities, while remaining transparent and extensible.'},
        ])

    print(f'Wrote {OUT}')


if __name__ == '__main__':
    build()
