# Football Match Result Prediction

Predicting Premier League match outcomes (home win / draw / away win)
from historical match data, and comparing the model against bookmaker odds.

## Status
Stage 1 in progress: loading, cleaning and exploring the data.

## Data
- Source: [football-data.co.uk](https://www.football-data.co.uk/englandm.php)
- Premier League seasons 2017/18 to 2025/26 (current season is incomplete)
- Raw files are not stored in this repo. To reproduce, download the CSVs
  into `data/raw/` and name them like `E1718.csv`.

## Project structure
- `notebooks/`: exploration and analysis
- `src/`: reusable code
- `data/processed/`: cleaned data

## Setup
    uv sync
    uv run jupyter lab

## Plan
1. Load and clean the data
2. Explore with charts and set a baseline
3. SQL analysis with DuckDB
4. Feature engineering (form, Elo ratings)
5. Models: logistic regression, gradient boosting
6. Compare against bookmaker probabilities
7. Streamlit demo

## Results
Coming soon.

## Limitations
Coming soon.