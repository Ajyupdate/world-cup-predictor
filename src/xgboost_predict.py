from pathlib import Path

import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
# The trained XGBoost bundle is saved under `data/models` with extension .pk1
MODEL_PATH = BASE_DIR / "data" / "models" / "xgboost_model.pk1"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"XGBoost model bundle not found at {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
encoder = bundle["encoder"]
features = bundle["features"]


sample = {
    "home_elo": 1920,
    "away_elo": 1800,
    "elo_diff": 120,
    "home_form_points": 12,
    "away_form_points": 7,
    "home_avg_goals": 2.1,
    "away_avg_goals": 1.3,
    "home_avg_conceded": 0.7,
    "away_avg_conceded": 1.2,
    "home_advantage": 1,
    "home_wins": 4,
    "away_wins": 2,

    "home_draws": 0,
    "away_draws": 1,

    "home_losses": 1,
    "away_losses": 2,
    "tournament_importance": 5,
}

# Build DataFrame and ensure it has the exact columns the model expects
match = pd.DataFrame([sample])

# Add any missing features (set to 0 by default) and drop unexpected ones
for f in features:
    if f not in match.columns:
        match[f] = 0

# Reorder to match training feature order
match = match[features]

probabilities = model.predict_proba(match)[0]

classes = encoder.classes_

print("\nPredicted probabilities:")
for outcome, prob in zip(classes, probabilities):
    print(f"{outcome}: {prob * 100:.2f}%")

