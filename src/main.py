import pandas as pd
from pathlib import Path
from elo import EloRatingSystem
from feature_builder import build_features
# Load CSV relative to this script (world-cup-predictor/data/matches.csv)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "results.csv"
matches = pd.read_csv(DATA_PATH)

features = build_features(matches)

features.to_csv(Path(__file__).resolve().parent.parent / "data" / "features.csv", index=False)

print("Feature dataset generated")

