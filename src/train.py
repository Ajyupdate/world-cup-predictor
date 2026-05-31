import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
df = pd.read_csv(DATA_DIR / "features.csv")

X = df[["home_elo", "away_elo", "elo_diff"]]

encoder = LabelEncoder()

y = encoder.fit_transform(df["target"])

model = LogisticRegression(
    max_iter=1000
)

model.fit(X, y)

print("Training complete")

predictions = model.predict(X)
accuracy = (predictions == y).mean()
print(f"Training accuracy: {accuracy:.2%}")

decoded_predictions = encoder.inverse_transform(predictions)

results = pd.DataFrame({
    "Actual": df["target"],
    "Predicted": decoded_predictions
})

print(results.head(10))