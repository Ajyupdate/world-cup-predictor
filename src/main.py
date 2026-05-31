import pandas as pd
from pathlib import Path
from elo import EloRatingSystem

# Load CSV relative to this script (world-cup-predictor/data/matches.csv)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "matches.csv"
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")

elo = EloRatingSystem()

for _, row in df.iterrows():
    elo.update_match(
        row["home_team"],
        row["away_team"],
        row["home_score"],
        row["away_score"]
    )

print("\nFinal Elo Ratings\n")

for team, rating in sorted(
    elo.ratings.items(),
    key=lambda x: x[1],
    reverse=True):
    print(f"{team}: {rating:.2f}")
