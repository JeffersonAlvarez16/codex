import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib


class SportsBettingModel:
    """Machine learning models for football betting markets."""

    def __init__(self, rolling_window: int = 5):
        self.rolling_window = rolling_window
        self.encoders = {}
        self.scaler = None
        self.models = {}

    def load_data(self, path: str) -> pd.DataFrame:
        """Load historical data from CSV."""
        df = pd.read_csv(path, parse_dates=['date'])
        df.sort_values('date', inplace=True)
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and encode data, generate rolling features."""
        df = df.copy()
        # Basic cleaning
        df.fillna(method='ffill', inplace=True)

        # Encode categorical variables
        for col in ['home_team', 'away_team']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le

        # Rolling averages of past performance
        stats_cols = ['home_goals', 'away_goals', 'home_shots', 'away_shots',
                      'home_corners', 'away_corners', 'home_xG', 'away_xG',
                      'home_possession', 'away_possession',
                      'home_cards', 'away_cards']
        for team_col in ['home_team', 'away_team']:
            for stat in stats_cols:
                if stat.startswith('home_'):
                    team_stat = stat.replace('home_', '')
                else:
                    team_stat = stat.replace('away_', '')
                new_col = f'{team_col}_{team_stat}_rolling'
                df[new_col] = (
                    df.groupby(team_col)[stat]
                    .shift()
                    .rolling(self.rolling_window)
                    .mean()
                )

        df.fillna(0, inplace=True)
        # Scale numerical features
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        self.scaler = StandardScaler()
        df[num_cols] = self.scaler.fit_transform(df[num_cols])
        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables and implied probabilities from odds."""
        df = df.copy()
        df['total_goals'] = df['home_goals'] + df['away_goals']
        df['result'] = df.apply(lambda x: 0 if x['home_goals'] > x['away_goals'] else (1 if x['home_goals'] == x['away_goals'] else 2), axis=1)
        df['ou25'] = (df['total_goals'] > 2.5).astype(int)
        df['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)
        for col in [
            'home_win_odds', 'draw_odds', 'away_win_odds',
            'over25_odds', 'under25_odds', 'btts_yes_odds', 'btts_no_odds'
        ]:
            if col in df:
                df[col.replace('_odds', '_implied_prob')] = 1 / df[col]
        return df

    def time_split(self, df: pd.DataFrame, test_size: float = 0.2):
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        return train_df, test_df

    def train_models(self, train_df: pd.DataFrame):
        features = [c for c in train_df.columns if c not in [
            'date', 'result', 'ou25', 'btts',
            'home_goals', 'away_goals', 'total_goals'
        ]]

        X = train_df[features]
        y1 = train_df['result']
        y2 = train_df['ou25']
        y3 = train_df['btts']

        self.models['1X2'] = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss')
        self.models['1X2'].fit(X, y1)

        self.models['OU25'] = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        self.models['OU25'].fit(X, y2)

        self.models['BTTS'] = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        self.models['BTTS'].fit(X, y3)

    def evaluate(self, test_df: pd.DataFrame):
        features = [c for c in test_df.columns if c not in [
            'date', 'result', 'ou25', 'btts',
            'home_goals', 'away_goals', 'total_goals'
        ]]
        X_test = test_df[features]
        y_test = {
            '1X2': test_df['result'],
            'OU25': test_df['ou25'],
            'BTTS': test_df['btts']
        }
        metrics = {}
        for market, model in self.models.items():
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)
            if market == '1X2':
                auc = roc_auc_score(y_test[market], probs, multi_class='ovo')
            else:
                auc = roc_auc_score(y_test[market], probs[:, 1])
            metrics[market] = {
                'accuracy': accuracy_score(y_test[market], preds),
                'precision': precision_score(y_test[market], preds, average='macro'),
                'recall': recall_score(y_test[market], preds, average='macro'),
                'f1': f1_score(y_test[market], preds, average='macro'),
                'auc': auc
            }
        return metrics

    def predict_match(self, match_df: pd.DataFrame) -> pd.DataFrame:
        df = match_df.copy()
        for col, le in self.encoders.items():
            df[col] = le.transform(df[col])
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[num_cols] = self.scaler.transform(df[num_cols])
        results = {}
        for market, model in self.models.items():
            probs = model.predict_proba(df)
            results[market] = probs
        return results

    def find_value_bets(self, match_df: pd.DataFrame) -> pd.DataFrame:
        predictions = self.predict_match(match_df)
        value_bets = []
        for idx, row in match_df.iterrows():
            res = {}
            if 'home_win_odds' in row:
                implied = 1 / row['home_win_odds']
                prob = predictions['1X2'][idx, 0]
                if prob > implied:
                    res['home_win'] = {'prob': prob, 'odds': row['home_win_odds']}
            if 'draw_odds' in row:
                implied = 1 / row['draw_odds']
                prob = predictions['1X2'][idx, 1]
                if prob > implied:
                    res['draw'] = {'prob': prob, 'odds': row['draw_odds']}
            if 'away_win_odds' in row:
                implied = 1 / row['away_win_odds']
                prob = predictions['1X2'][idx, 2]
                if prob > implied:
                    res['away_win'] = {'prob': prob, 'odds': row['away_win_odds']}
            if 'over25_odds' in row:
                implied = 1 / row['over25_odds']
                prob = predictions['OU25'][idx, 1]
                if prob > implied:
                    res['over25'] = {'prob': prob, 'odds': row['over25_odds']}
            if 'under25_odds' in row:
                implied = 1 / row['under25_odds']
                prob = 1 - predictions['OU25'][idx, 1]
                if prob > implied:
                    res['under25'] = {'prob': prob, 'odds': row['under25_odds']}
            if 'btts_yes_odds' in row:
                implied = 1 / row['btts_yes_odds']
                prob = predictions['BTTS'][idx, 1]
                if prob > implied:
                    res['btts_yes'] = {'prob': prob, 'odds': row['btts_yes_odds']}
            if 'btts_no_odds' in row:
                implied = 1 / row['btts_no_odds']
                prob = 1 - predictions['BTTS'][idx, 1]
                if prob > implied:
                    res['btts_no'] = {'prob': prob, 'odds': row['btts_no_odds']}
            value_bets.append(res)
        return pd.DataFrame(value_bets)


def batch_predict(model: SportsBettingModel, upcoming_matches_path: str) -> pd.DataFrame:
    df = pd.read_csv(upcoming_matches_path, parse_dates=['date'])
    return model.find_value_bets(df)


def save_results(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sports betting ML tool")
    parser.add_argument("data", help="CSV file with historical match data")
    parser.add_argument("upcoming", help="CSV file with upcoming matches")
    parser.add_argument("--window", type=int, default=5, help="Rolling window for stats")
    parser.add_argument("--output", default="predictions.csv", help="Where to save predictions")
    args = parser.parse_args()

    model = SportsBettingModel(rolling_window=args.window)
    data = model.load_data(args.data)
    data = model.preprocess(data)
    data = model.feature_engineering(data)
    train_df, test_df = model.time_split(data)
    model.train_models(train_df)
    metrics = model.evaluate(test_df)
    print("Evaluation:", metrics)

    preds = batch_predict(model, args.upcoming)
    save_results(preds, args.output)
    print(f"Predictions saved to {args.output}")
