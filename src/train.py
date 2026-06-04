import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
df = pd.read_csv(DATA_DIR / "features.csv")

# Drop future/unplayed matches that have no target label
# and sort chronologically by date so training uses older matches.
df = df[df["target"].notna()].copy()
df["data"] = pd.to_datetime(df["data"], errors="coerce")
df = df.sort_values("data").reset_index(drop=True)

#FEATURES
feature_columns = [
    "home_elo",
    "away_elo",
    "elo_diff",

    "home_form_points",
    "away_form_points",

    "home_avg_goals",
    "away_avg_goals",

    "home_avg_conceded",
    "away_avg_conceded",

    "home_wins",
    "away_wins",
    "home_draws",
    "away_draws",
    "home_losses",
    "away_losses"
]
X = df[feature_columns]

#TARGET

encoder = LabelEncoder()
df = df.dropna(subset=["target"]).copy()
y = encoder.fit_transform(df["target"])

#TRAIN TEST SPLIT

split_index = int(len(df) * 0.8)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

#MODEL
model = LogisticRegression(
    max_iter=5000
)

model.fit(X_train, y_train)

#EVALUATION

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Test accuracy: {accuracy:.4%}")


print(classification_report(y_test, predictions, target_names=encoder.classes_))

#SAVE MODEL
model_dir = DATA_DIR / "models"
model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_dir / "logistic_model.pk1")
joblib.dump(encoder, model_dir / "label_encoder.pk1")

print("Model and encoder saved")