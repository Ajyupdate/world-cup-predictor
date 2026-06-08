from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "data" / "models" / "xgboost_model.pk1"

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]

features = bundle["features"]

importance = model.feature_importances_

df = pd.DataFrame({"feature": features, "importance": importance})

df = df.sort_values("importance", ascending=False)

print(df)


