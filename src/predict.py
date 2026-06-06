import joblib
from pathlib import Path
import pandas as pd

# Models are saved under the project's `data/models` directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
model_path = DATA_DIR / "models" / "logistic_model.pk1"
encoder_path = DATA_DIR / "models" / "label_encoder.pk1"

if not model_path.exists() or not encoder_path.exists():
    raise FileNotFoundError(
        f"Model or encoder not found. Expected: {model_path} and {encoder_path}"
    )

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

sample_match = pd.DataFrame([{
    "home_elo": 1850,
    "away_elo": 1750,

    "elo_diff": 100,

    "home_form_points": 12,
    "away_form_points": 6,

    "home_avg_goals": 2.2,
    "away_avg_goals": 1.1,

    "home_avg_conceded": 0.7,
    "away_avg_conceded": 1.5,

    "home_wins": 4,
    "away_wins": 2,

    "home_draws": 0,
    "away_draws": 1,

    "home_losses": 1,
    "away_losses": 2

}])
# prediction = model.predict(sample_match)

# result = encoder.inverse_transform(prediction)
print("Classes:", encoder.classes_)
proba = model.predict_proba(sample_match)[0]
for cls, p in zip(encoder.classes_, proba):
    print(f"{cls}: {p:.2%}")
predicted = encoder.inverse_transform([proba.argmax()])[0]
print("Predicted:", predicted)