from pathlib import Path
import pandas as pd
from feature_builder import build_features
from xgboost_predict import predict_probabilities

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "results.csv"
df = pd.read_csv(DATA_PATH)

features_df = build_features(df)

future_matches = features_df[
    features_df["target"].isna()
]

print("\nWORLD CUP PREDICTIONS\n")

predictions = []
for _, row in future_matches.iterrows():
    feature_row = row.drop(
        ["date", "home_team", "away_team", "target"],
        errors="ignore"
    ).to_dict()

    probs = predict_probabilities(feature_row)

    # print(f"{row['home_team']} vs {row['away_team']}")
    # print(f"HOME_WIN: {probs['HOME_WIN']*100:.2f}%")
    # print(
    #     f"DRAW: "
    #     f"{probs['DRAW']*100:.2f}%"
    # )

    # print(
    #     f"AWAY_WIN: "
    #     f"{probs['AWAY_WIN']*100:.2f}%"
    # )

    # print("-" * 40)
    
    predictions.append({
        "home_team": row['home_team'],
        "away_team": row['away_team'],
        "home_win_probability": probs['HOME_WIN'] * 100,
        "draw_probability": probs['DRAW'] * 100,
        "away_win_probability": probs['AWAY_WIN'] * 100
    })
    
pd.DataFrame(predictions).to_csv("world_cup_predictions.csv", index=False)



