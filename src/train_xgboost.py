from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import(
    accuracy_score,
    log_loss,
)

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "features.csv"
MODEL_PATH = BASE_DIR / "data" / "models" / "xgboost_model.pk1"

def main():
    df = pd.read_csv(DATA_PATH)

    # Drop future/unplayed matches and sort chronologically
    df = df[df["target"].notna()].copy()
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.sort_values("data").reset_index(drop=True)

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
        "away_losses",

        "home_advantage",
        "tournament_importance"
    ]
    X = df[feature_columns]
    encoder = LabelEncoder()

    y = encoder.fit_transform(df["target"])   

    split_index = int(len(df) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,

        n_estimators=300,
        max_depth=5,

        learning_rate=0.05,

        subsample=0.8,
        colsample_bytree=0.8,

        random_state=42
    ) 

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probailities = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)

    loss = log_loss(y_test, probailities)

    print(f"Test accuracy: {accuracy:.4%}")
    print(f"Log loss: {loss:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": model,
        "encoder": encoder,
        "features": feature_columns
    },
        MODEL_PATH
    )
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
