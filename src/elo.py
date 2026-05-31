from collections import defaultdict

K_FACTOR = 20

class EloRatingSystem:
    def __init__(self):
        self.ratings = defaultdict(lambda: 1500)

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_match(self, home_team, away_team, home_goals, away_goals):
        

        home_rating = self.ratings[home_team]
        away_rating = self.ratings[away_team]
        
        expected_home = self.expected_score(home_rating, away_rating)
        expected_away = self.expected_score(away_rating, home_rating)

        if home_goals > away_goals:
            actual_home = 1
            actual_away = 0
        elif home_goals < away_goals:
            actual_home = 0
            actual_away = 1
        else:
            actual_home = 0.5
            actual_away = 0.5

        new_home_rating = home_rating + K_FACTOR * (actual_home - expected_home)
        new_away_rating = away_rating + K_FACTOR * (actual_away - expected_away)    

        self.ratings[home_team] = new_home_rating
        self.ratings[away_team] = new_away_rating