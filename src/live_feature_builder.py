from feature_builder import calculate_form
def build_match_features(
    home_team,
    away_team,
    elo_system,
    histories
):

    home_elo = elo_system.ratings[
        home_team
    ]

    away_elo = elo_system.ratings[
        away_team
    ]

    home_form = calculate_form(
        histories[home_team]
    )

    away_form = calculate_form(
        histories[away_team]
    )

    return {
        "home_elo": home_elo,
        "away_elo": away_elo,

        "elo_diff":
            home_elo - away_elo,

        "home_form_points":
            home_form["form_points"],

        "away_form_points":
            away_form["form_points"],

        "home_avg_goals":
            home_form["avg_goals_scored"],

        "away_avg_goals":
            away_form["avg_goals_scored"],

        "home_avg_conceded":
            home_form["avg_goals_conceded"],

        "away_avg_conceded":
            away_form["avg_goals_conceded"],

        "home_wins":
            home_form["wins"],

        "home_draws":
            home_form["draws"],

        "home_losses":
            home_form["losses"],

        "away_wins":
            away_form["wins"],

        "away_draws":
            away_form["draws"],

        "away_losses":
            away_form["losses"],

        "home_advantage": 0,

        "tournament_importance": 5
    }