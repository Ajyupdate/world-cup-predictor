from pathlib import Path

import pandas as pd
from elo import EloRatingSystem

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def build_features(df):
    elo = EloRatingSystem()
    rows = []

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_elo = elo.ratings[home_team]
        away_elo = elo.ratings[away_team]

        if row["home_score"] > row["away_score"]:
            target = "HOME_WIN"

        elif row["home_score"] < row["away_score"]:
            target = "AWAY_WIN"

        else:
            target = "DRAW"

        rows.append({
            "home_team": home_team,
            "away_team": away_team,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "target": target
        })

        elo.update_match(
            home_team,
            away_team,
            row["home_score"],
            row["away_score"]
        )

    return pd.DataFrame(rows)

df = pd.read_csv(DATA_DIR / "matches.csv")
features_df = build_features(df)

features_df.to_csv(DATA_DIR / "features.csv", index=False)