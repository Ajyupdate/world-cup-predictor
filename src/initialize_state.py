from collections import defaultdict, deque
from pathlib import Path
import pandas as pd

from elo import EloRatingSystem

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "results.csv"

def initialize_state():
    df = pd.read_csv(DATA_FILE)

    df = df.sort_values("date")

    elo = EloRatingSystem()

    team_histories = defaultdict(
        lambda: deque(maxlen=5)
    )
    
    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_score = row["home_score"]
        away_score = row["away_score"]
        
        home_result = (
            "W" if home_score > away_score else
            "L" if home_score < away_score else
            "D"
        )
        
        away_result = (
            "W" if away_score > home_score else
            "L" if away_score < home_score else
            "D"
        )
        
        team_histories[home_team].append({
            "goals_scored": home_score,
            "goals_conceded": away_score,
            "result": home_result
        })
        
        team_histories[away_team].append({
            "goals_scored": away_score,
            "goals_conceded": home_score,
            "result": away_result
        })

        elo.update_match(
            home_team,
            away_team,
            home_score,
            away_score
        )
    
    return elo, team_histories
    