from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
from elo import EloRatingSystem

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TOURNAMENT_WEIGHTS = {
    "Friendly": 1.0,

    "UEFA NATIONS LEAGUE": 2,

    "World Cup qualification": 3,

    "UEFA Euro qualification": 3,

    "Copa America": 4,

    "UEFA Euro": 4,

    "FIFA World Cup": 5
}

def get_tournament_importance(tournament):
    return TOURNAMENT_WEIGHTS.get(tournament, 2)
def calculate_form(history):
    """
    Calculate recent form statistics from the last N matches
    """
    if len(history) == 0:
        return {
            "form_points": 0,
            "avg_goals_scored": 0,
            "avg_goals_conceded": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0
        }
    points = 0
    goals_scored = 0
    goals_conceded = 0

    wins = 0
    draws = 0
    losses = 0

    for match in history:
        goals_scored += match["goals_scored"]
        goals_conceded += match["goals_conceded"]
        if match["result"] == "W":
            points += 3
            wins += 1
        elif match["result"] == "D":
            points += 1
            draws += 1
        else:
            losses += 1
    total_matches = len(history)
    return {
        "form_points": points,
        "avg_goals_scored": goals_scored / total_matches,
        "avg_goals_conceded": goals_conceded / total_matches,
        "wins": wins,
        "draws": draws,
        "losses": losses
    }

def get_result(home_score, away_score):
    """
    Convert scoreline to target label
    """
    if home_score > away_score:
        return "HOME_WIN"
    elif home_score < away_score:
        return "AWAY_WIN"
    else:
        return "DRAW"
            

def build_features(df):
    """
    Build machine learning dataset
    """
    #sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    elo = EloRatingSystem()

    #store last 5 matches per team
    team_histories = defaultdict(lambda: deque(maxlen=5))
    rows = []

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_score = row["home_score"]
        away_score = row["away_score"]

        # Determine whether the match has been played (scores present)
        is_played = not (pd.isna(home_score) or pd.isna(away_score))

        #ELO BEFORE MATCH

        home_elo = elo.ratings[home_team]
        away_elo = elo.ratings[away_team]

        #FORM BEFORE MATCH

        home_form = calculate_form(team_histories[home_team])
        away_form = calculate_form(team_histories[away_team])

        neutral = row["neutral"]
        home_advantage = 0 if neutral else 1

        importance = get_tournament_importance(row["tournament"])
        #TARGET (only if match played)

        target = get_result(home_score, away_score) if is_played else None

        #FEATURE ROW

        rows.append({
            "data": row["date"],

            "home_team": home_team,
            "away_team": away_team,
            "home_elo": home_elo,
            "away_elo": away_elo,

            "elo_diff": round(home_elo - away_elo, 2),

            #Form points
            "home_form_points": home_form["form_points"],
            "away_form_points": away_form["form_points"],

            #Goals scored
            "home_avg_goals": round(home_form["avg_goals_scored"], 2),
            "away_avg_goals": round(away_form["avg_goals_scored"], 2),

            #Goals conceded
            "home_avg_conceded": round(home_form["avg_goals_conceded"], 2),
            "away_avg_conceded": round(away_form["avg_goals_conceded"], 2),

            #WDL stats
            "home_wins": home_form["wins"],
            "home_draws": home_form["draws"],
            "home_losses": home_form["losses"],
            "away_wins": away_form["wins"],
            "away_draws": away_form["draws"],
            "away_losses": away_form["losses"],
            "home_advantage": home_advantage,
            "tournament_importance": importance,
            
            "target": target

        })

        #UPDATE TEAM HISTORIES
        #AFTER FEATURE CREATION
        #Only update histories and ELO for matches that were actually played
        if is_played:
            home_result = (
                "W"
                if home_score > away_score
                else "L"
                if home_score < away_score
                else "D"
            )

            away_result = (
                "W"
                if away_score > home_score
                else "L"
                if away_score < home_score
                else "D"
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

            #UPDATE ELO RATINGS
            #AFTER FEATURE CREATION

            elo.update_match(
                home_team, away_team, home_score, away_score
            )



       
    return pd.DataFrame(rows)

df = pd.read_csv(DATA_DIR / "results.csv")
features_df = build_features(df)

features_df.to_csv(DATA_DIR / "features.csv", index=False)