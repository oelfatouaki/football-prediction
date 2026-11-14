# Premier League Match Result Prediction

## Project Overview

The goal of this project is to predict the outcome of Premier League matches
(home win, draw, or away win) using historical match data, and to compare the
model's predictions against bookmaker odds. The project covers the full data
science workflow: data cleaning, exploration, SQL analysis, feature
engineering, modeling, evaluation, and a small web demo.

## About Me

Hi, I'm Oussama, a Computer Science graduate (Bac +2) learning data
science and AI. I'm building real-world projects in Python, SQL and machine
learning to prepare for university studies and a career in data.

- Languages and tools: Python, SQL, Git
- Interests: sports analytics, machine learning, building things end to end
- Currently learning: feature engineering, model evaluation, deployment

## Objectives

- Clean and combine multiple seasons of match data
- Explore trends such as home advantage and goals per game
- Analyze the data with SQL (DuckDB), including rolling form and streaks
- Engineer features: recent form, goals scored and conceded, Elo ratings
- Train and compare models against a simple baseline
- Benchmark the best model against bookmaker-implied probabilities
- Present the results in an interactive Streamlit app

## Models

- **Baseline:** always predict a home win
- **Logistic Regression**
- **Gradient Boosting:** LightGBM
- **Optional:** Poisson goals model

## Technologies Used

- Python 3.12
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn, LightGBM
- DuckDB (SQL)
- Streamlit
- uv, Git/GitHub, Cursor

## Data

- Source: [football-data.co.uk](https://www.football-data.co.uk/englandm.php)
- Premier League seasons 2017/18 onward (the current season is incomplete)
- Includes results, shots, fouls, cards, corners and bookmaker odds
- Raw files are not stored in this repository. To reproduce the project,
  download the CSVs into `data/raw/` and name them like `E1718.csv`.

## Data Processing

- Combined all seasons into one dataset
- Parsed mixed date formats (`dayfirst=True`)
- Checked for duplicates, missing values and inconsistent team names
- Sorted matches chronologically
- Saved the cleaned data to `data/processed/matches.csv`

## Validation Approach

- **Time-based split:** train on older seasons, test on newer ones
- **No data leakage:** features use only information available before each match
- **Metrics:** accuracy and log loss
- **Benchmarks:** the "always home win" baseline and bookmaker odds

## Results

| Model                    | Accuracy | Log Loss |
|--------------------------|----------|----------|
| Baseline (always home)   | TBD      | TBD      |
| Logistic Regression      | TBD      | TBD      |
| LightGBM                 | TBD      | TBD      |
| Bookmaker odds           | TBD      | TBD      |

Results will be added once the models are trained and evaluated.

## Project Structure

```
├── data/
│   ├── raw/           # downloaded CSVs (not tracked)
│   └── processed/     # cleaned data
├── notebooks/         # exploration and modeling
├── src/               # reusable code
├── app.py             # Streamlit demo (planned)
├── pyproject.toml
├── README.md
```

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/football-prediction
cd football-prediction
uv sync
uv run jupyter lab
```

Download the data into `data/raw/` as described above, then run the
notebooks in order.

## Limitations

To be completed after the analysis. Expected topics: small sample of seasons,
no player-level or injury data, and the difficulty of beating bookmaker odds.

## Future Improvements

- Add expected goals (xG) data
- Try a Poisson/Dixon-Coles goals model
- Add more leagues
- Improve model explainability
- Deploy the live demo