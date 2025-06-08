# Sports Betting Machine Learning Tool

This repository contains a modular Python tool for building and evaluating machine learning models on historical football match data. It can predict outcomes for common betting markets and identify potential value bets based on model probabilities versus bookmaker odds.

## Features
- Load data from CSV files
- Preprocess and encode categorical columns
- Generate rolling statistics for each team
- Train XGBoost models for the following markets:
  - 1X2 (home/draw/away)
  - Over/Under 2.5 goals
  - Both Teams To Score
- Evaluate models with accuracy, precision, recall, F1-score and AUC
- Predict upcoming matches and flag value betting opportunities
- Batch prediction and result export to CSV

## Requirements
See `requirements.txt` for the list of Python dependencies. Install them via:

```bash
pip install -r requirements.txt
```

## Usage
Train the models and generate predictions for upcoming matches:

```bash
python sports_betting_tool.py historical_data.csv upcoming_matches.csv --window 5 --output predictions.csv
```

`historical_data.csv` should contain columns such as:

- `date` – match date
- `home_team`, `away_team`
- `home_goals`, `away_goals`
- `home_shots`, `away_shots`, `home_corners`, `away_corners`
- `home_xG`, `away_xG`, `home_possession`, `away_possession`
- `home_cards`, `away_cards`
- Odds columns: `home_win_odds`, `draw_odds`, `away_win_odds`, `over25_odds`, `under25_odds`, `btts_yes_odds`, `btts_no_odds`

`upcoming_matches.csv` should contain the same feature columns (without the result columns).

The script outputs a CSV file with predicted probabilities and suggested value bets for each match.

## Automatic Training Example
The script `auto_betting_tool.py` downloads recent seasons from the
[openfootball](https://github.com/openfootball/football.json) dataset.
Specify the competition code with `--competition` (e.g. `en.1` for the Premier
League, `es.1` for La Liga). Team names **must** be written exactly as in the
dataset (usually in English). If you request international matches, use the
appropriate competition code like `worldcup`. Example:

```bash
python auto_betting_tool.py "Manchester United" "Chelsea" 2021-05-01 --competition en.1
```

For international matches:

```bash
python auto_betting_tool.py "Germany" "France" 2018-06-15 --competition worldcup
```

The tool fetches recent seasons from the [openfootball](https://github.com/openfootball/football.json) dataset, trains models and prints probabilities for 1X2, Over 2.5, BTTS and sample parlays.
