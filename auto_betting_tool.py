import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

SEASON_URL = (
    "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{competition}.json"
)


def download_season_data(year: int, competition: str) -> pd.DataFrame:
    """Download a single season for the given competition code."""
    season = f"{year}-{str(year + 1)[-2:]}"
    url = SEASON_URL.format(season=season, competition=competition)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = []
    for m in data.get("matches", []):
        if "score" not in m or "ft" not in m["score"]:
            continue
        rows.append({
            "date": pd.to_datetime(m["date"]),
            "home_team": m["team1"],
            "away_team": m["team2"],
            "home_goals": m["score"]["ft"][0],
            "away_goals": m["score"]["ft"][1],
        })
    return pd.DataFrame(rows)

def load_historical_data(
    match_date: pd.Timestamp, competition: str, seasons_back: int = 2
) -> pd.DataFrame:
    """Load recent seasons for the requested competition."""
    dfs = []
    year = match_date.year - (1 if match_date.month < 7 else 0)
    for i in range(seasons_back + 1):
        dfs.append(download_season_data(year - i, competition))
    df = pd.concat(dfs, ignore_index=True)
    df.sort_values("date", inplace=True)
    return df

def preprocess(df: pd.DataFrame, window: int = 5):
    df = df.copy()
    encoders = {}
    for col in ["home_team", "away_team"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    for team_col, gf_col, ga_col in [("home_team", "home_goals", "away_goals"), ("away_team", "away_goals", "home_goals")]:
        df[f"{team_col}_gf_avg"] = df.groupby(team_col)[gf_col].shift().rolling(window).mean()
        df[f"{team_col}_ga_avg"] = df.groupby(team_col)[ga_col].shift().rolling(window).mean()
    df.fillna(0, inplace=True)
    df["result"] = df.apply(lambda x: 0 if x["home_goals"] > x["away_goals"] else (1 if x["home_goals"] == x["away_goals"] else 2), axis=1)
    df["ou25"] = ((df["home_goals"] + df["away_goals"]) > 2.5).astype(int)
    df["btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)
    scaler = StandardScaler()
    feature_cols = df.select_dtypes(include=[np.number]).columns.difference([
        "home_goals",
        "away_goals",
        "result",
        "ou25",
        "btts",
    ])
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, encoders, scaler, list(feature_cols)

def train_models(df: pd.DataFrame):
    feats = [c for c in df.columns if c not in ["date", "home_goals", "away_goals", "result", "ou25", "btts"]]
    X = df[feats]
    models = {
        "1X2": XGBClassifier(objective="multi:softprob", eval_metric="mlogloss"),
        "OU25": XGBClassifier(objective="binary:logistic", eval_metric="logloss"),
        "BTTS": XGBClassifier(objective="binary:logistic", eval_metric="logloss"),
    }
    models["1X2"].fit(X, df["result"])
    models["OU25"].fit(X, df["ou25"])
    models["BTTS"].fit(X, df["btts"])
    return models, feats

def make_features(info: dict, df: pd.DataFrame, window: int, encoders, scaler, feature_cols, competition: str):
    missing = [t for t in [info["home_team"], info["away_team"]]
               if t not in encoders["home_team"].classes_]
    if missing:
        examples = ", ".join(sorted(encoders["home_team"].classes_[:5]))
        raise ValueError(
            "Team{} {} not found in historical data for competition '{}'. "
            "Check the competition code and use the English name as in the "
            "dataset (e.g. 'Germany' instead of 'Alemania'). Available "
            "teams include: {}...".format(
                "s" if len(missing) > 1 else "",
                ", ".join(missing),
                competition,
                examples,
            )
        )
    temp = df[df["date"] < info["date"]].copy()
    for team_col, gf_col, ga_col in [("home_team", "home_goals", "away_goals"), ("away_team", "away_goals", "home_goals")]:
        temp[f"{team_col}_gf_avg"] = temp.groupby(team_col)[gf_col].shift().rolling(window).mean()
        temp[f"{team_col}_ga_avg"] = temp.groupby(team_col)[ga_col].shift().rolling(window).mean()
    last = temp.tail(1)
    row = {
        "home_team": info["home_team"],
        "away_team": info["away_team"],
        "home_team_gf_avg": last[last["home_team"] == info["home_team"]]["home_team_gf_avg"].values[-1] if not last[last["home_team"] == info["home_team"]].empty else 0,
        "home_team_ga_avg": last[last["home_team"] == info["home_team"]]["home_team_ga_avg"].values[-1] if not last[last["home_team"] == info["home_team"]].empty else 0,
        "away_team_gf_avg": last[last["away_team"] == info["away_team"]]["away_team_gf_avg"].values[-1] if not last[last["away_team"] == info["away_team"]].empty else 0,
        "away_team_ga_avg": last[last["away_team"] == info["away_team"]]["away_team_ga_avg"].values[-1] if not last[last["away_team"] == info["away_team"]].empty else 0,
    }
    feat_df = pd.DataFrame([row])
    for col, le in encoders.items():
        feat_df[col] = le.transform(feat_df[col])
    feat_df.fillna(0, inplace=True)
    feat_df[feature_cols] = scaler.transform(feat_df[feature_cols])
    return feat_df[feature_cols]

def predict(home: str, away: str, date: str, competition: str, window: int = 5):
    mdate = pd.to_datetime(date)
    hist = load_historical_data(mdate, competition)
    prep, encoders, scaler, feat_cols = preprocess(hist, window)
    models, feats = train_models(prep)
    match_feats = make_features({'home_team': home, 'away_team': away, 'date': mdate}, hist, window, encoders, scaler, feat_cols, competition)
    probs = {k: m.predict_proba(match_feats[feats]) for k, m in models.items()}
    parlay_home_over = float(probs['1X2'][0, 0] * probs['OU25'][0, 1])
    parlay_away_over = float(probs['1X2'][0, 2] * probs['OU25'][0, 1])
    return probs, {'home_win_and_over25': parlay_home_over, 'away_win_and_over25': parlay_away_over}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Auto sports betting ML tool')
    parser.add_argument('home', help='Home team')
    parser.add_argument('away', help='Away team')
    parser.add_argument('date', help='Match date YYYY-MM-DD')
    parser.add_argument(
        '--competition',
        default='en.1',
        help='Openfootball competition code (e.g. en.1 for Premier League)',
    )
    parser.add_argument('--window', type=int, default=5, help='Rolling window')
    args = parser.parse_args()
    prob, parlays = predict(
        args.home,
        args.away,
        args.date,
        args.competition,
        args.window,
    )
    print('Probabilities:')
    print('1X2:', prob['1X2'][0])
    print('Over 2.5:', prob['OU25'][0, 1])
    print('BTTS:', prob['BTTS'][0, 1])
    print('Parlays:', parlays)
